import torch
import csv
import os

import numpy as np
import argparse
from dataloader_category_type import PoiLoader, Split, Usage
from torch.utils.data import DataLoader
from setting import Setting
from trainer_category_type_CL_transformer import FlashbackPlusPlusTrainer
from network_category_type_CL_transformer import create_h0_strategy
import pickle

with open('CAL_locationReindex.pkl', 'rb') as file:  # 'rb' 模式表示以二进制读取文件
    POIReindex = pickle.load(file)

with open('CAL_individual_locationReindex.pkl', 'rb') as file:  # 'rb' 模式表示以二进制读取文件
    locationReindex = pickle.load(file)

with open('POI_highest_star_location_CAL.pkl', 'rb') as file:  # 'rb' 模式表示以二进制读取文件
    POI_highest_star_location = pickle.load(file)


reindexed_match_poi_highest_score_location = {}

for poi_id_initial, location_id_initial in POI_highest_star_location.items():

    poi_id_reindex = POIReindex.get(poi_id_initial, None)
    location_id_reindex = locationReindex.get(location_id_initial, None)

    if poi_id_reindex is not None and location_id_reindex is not None:
        reindexed_match_poi_highest_score_location[poi_id_reindex] = location_id_reindex

def update_weights(epoch, total_epochs, initial_weights, final_weights):
    alpha = epoch / total_epochs
    return [(1 - alpha) * iw + alpha * fw for iw, fw in zip(initial_weights, final_weights)]

# early stopping
patience = 20  # 

### parse settings ###
setting = Setting()
setting.parse()
print(setting,'setting')


### load dataset ###
poi_loader = PoiLoader(setting.max_users, setting.min_checkins,setting.sequence_length)
poi_loader.load(setting.dataset_file)
dataset = poi_loader.poi_dataset(setting.sequence_length, setting.batch_size, Split.TRAIN, Usage.MAX_SEQ_LENGTH)
dataset_test = poi_loader.poi_dataset(setting.sequence_length, setting.batch_size, Split.TEST, Usage.MAX_SEQ_LENGTH)
dataloader = DataLoader(dataset, batch_size = 1, shuffle=False,drop_last=True)
dataloader_test = DataLoader(dataset_test, batch_size = 1, shuffle=False,drop_last=True)

# setup trainer
seed = 2024
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
import random
random.seed(seed)
trainer = FlashbackPlusPlusTrainer(setting.lambda_t, setting.lambda_s,setting.alpha_start, setting.alpha_end, setting.epochs,setting.decay_type,setting.weight_type)

trainer.prepare(poi_loader.locations(), poi_loader.individual_locations(),poi_loader.category(),poi_loader.location_type(),poi_loader.user_count(), setting.hidden_dim, setting.rnn_factory, setting.device,setting)
h0_strategy = create_h0_strategy(setting.hidden_dim, setting.is_lstm)
print('{} {}'.format(trainer.greeter(), setting.rnn_factory.greeter()))

optimizer = torch.optim.Adam(trainer.parameters(), lr = setting.learning_rate, weight_decay = setting.weight_decay)
scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.95)

# review function
def review_mode(complex_samples, trainer, optimizer, tau, thre, poi_weight, cate_weight, type_weight):
    revisit_samples = random.sample(complex_samples, setting.batch_size)
    x, c, type, t, s,y_t, y_s, y, c_y,type_y, active_users, h, c_h, type_h, *_ = zip(*revisit_samples)

    x = torch.stack(x, dim=1)
    c = torch.stack(c, dim=1)
    type = torch.stack(type, dim=1)
    t = torch.stack(t, dim=1)
    s = torch.stack(s, dim=1)
    y_t = torch.stack(y_t, dim=1)
    y_s = torch.stack(y_s, dim=1)
    y = torch.stack(y, dim=1)
    c_y = torch.stack(c_y, dim=1)
    type_y = torch.stack(type_y, dim=1)
    active_users = torch.stack([u.unsqueeze(0) if u.dim() == 0 else u for u in active_users], dim=1)

    if setting.is_lstm:
        h = (torch.stack([h0 for h0, _ in h], dim=1).to(setting.device),
             torch.stack([h1 for _, h1 in h], dim=1).to(setting.device))
        c_h = (torch.stack([ch0 for ch0, _ in c_h], dim=1).to(setting.device),
               torch.stack([ch1 for _, ch1 in c_h], dim=1).to(setting.device))
        type_h = (torch.stack([th0 for th0, _ in type_h], dim=1).to(setting.device),
                  torch.stack([th1 for _, th1 in type_h], dim=1).to(setting.device))
    else:
        h = torch.stack(h, dim=1).to(setting.device)
        c_h = torch.stack(c_h, dim=1).to(setting.device)
        type_h = torch.stack(type_h, dim=1).to(setting.device)

    optimizer.zero_grad()
    loss, cate_loss, type_loss, poi_difficulty, cate_difficulty, type_difficulty, *_ = trainer.loss(
        x, c, type, t, s, y, c_y,type_y,y_t, y_s, h, c_h, type_h, active_users
    )
    # adjust weight
    poi_difficulty_norm = (poi_difficulty - poi_difficulty.min()) / (poi_difficulty.max() - poi_difficulty.min())
    cate_difficulty_norm = (cate_difficulty - cate_difficulty.min()) / (cate_difficulty.max() - cate_difficulty.min())
    type_difficulty_norm = (type_difficulty - type_difficulty.min()) / (type_difficulty.max() - type_difficulty.min())
    poi_cate_diff = poi_weight * poi_difficulty_norm + cate_weight * cate_difficulty_norm + type_weight * type_difficulty_norm
    difficulty1 = 2 * (poi_cate_diff - poi_cate_diff.min().item()) / (poi_cate_diff.max().item() - poi_cate_diff.min().item()) - 1
    #w_j = torch.exp(-tau * (difficulty1 - thre) ** 2)
    #w_j = torch.exp(-tau * (difficulty1 - (torch.mean(difficulty1))) ** 2)
    w_j_poi = torch.exp(-tau * (poi_difficulty_norm - thre) ** 2)
    #gaosi_loss = torch.mean(w_j * loss)
    gaosi_loss = torch.mean(w_j_poi * loss)
    #loss = gaosi_loss + cate_loss + type_loss
    loss = gaosi_loss
    loss.backward()
    optimizer.step()


def evaluate_test():
    results = []
    dataset_test.reset()
    h = h0_strategy.on_init(setting.batch_size, setting.device)
    c_h = h0_strategy.on_init(setting.batch_size, setting.device)
    type_h = h0_strategy.on_init(setting.batch_size, setting.device)
    
    with torch.no_grad():        
        iter_cnt = 0
        recall1 = 0
        recall5 = 0
        recall10 = 0
        mrr1=0
        mrr5=0
        mrr10=0
        average_precision = 0.
        
        
        u_iter_cnt = np.zeros(poi_loader.user_count())
        
        u_recall1 = np.zeros(poi_loader.user_count())
        u_recall5 = np.zeros(poi_loader.user_count())
        u_recall10 = np.zeros(poi_loader.user_count())
        u_mrr1 = np.zeros(poi_loader.user_count())
        u_mrr5 = np.zeros(poi_loader.user_count())
        u_mrr10 = np.zeros(poi_loader.user_count())
        u_average_precision = np.zeros(poi_loader.user_count()) 
       

        reset_count = torch.zeros(poi_loader.user_count())
        
        for i, (x, c, type, t, s, y,individual_y,c_y, type_y, y_t, y_s, reset_h,c_reset_h,type_reset_h, active_users) in enumerate(dataloader_test):
            active_users = active_users.squeeze()
            for j, reset in enumerate(reset_h):
                if reset:
                    if setting.is_lstm:
                        hc = h0_strategy.on_reset_test(active_users[j], setting.device)
                        h[0][0, j] = hc[0]
                        h[1][0, j] = hc[1]
                    else:
                        h[0, j] = h0_strategy.on_reset_test(active_users[j], setting.device)
                    reset_count[active_users[j]] += 1
            for j, reset in enumerate(c_reset_h):
                if reset:
                    if setting.is_lstm:
                        c_hc = h0_strategy.on_reset_test(active_users[j], setting.device)
                        c_h[0][0, j] = c_hc[0]
                        c_h[1][0, j] = c_hc[1]
                    else:
                        c_h[0, j] = h0_strategy.on_reset_test(active_users[j], setting.device)

            for j, reset in enumerate(type_reset_h):
                if reset:
                    if setting.is_lstm:
                        type_hc = h0_strategy.on_reset_test(active_users[j], setting.device)
                        type_h[0][0, j] = type_hc[0]
                        type_h[1][0, j] = type_hc[1]
                    else:
                        type_h[0, j] = h0_strategy.on_reset_test(active_users[j], setting.device)
            
        
            x = x.squeeze().to(setting.device)
            c = c.squeeze().to(setting.device)
            type = type.squeeze().to(setting.device)
            t = t.squeeze().to(setting.device)
            s = s.squeeze().to(setting.device)            
            y = y.squeeze()
            individual_y = individual_y.squeeze()
            c_y = c_y.squeeze()
            type_y = type_y.squeeze()
            y_t = y_t.squeeze().to(setting.device)
            y_s = y_s.squeeze().to(setting.device)
            
            active_users = active_users.to(setting.device)            
        
            # evaluate:
            out, c_out,type_out,h, c_h ,type_h= trainer.evaluate(x, c,type, t, s, y_t, y_s, h,c_h,type_h, active_users)
            
            for j in range(setting.batch_size):  
                o = out[j]
                c_o=c_out[j]
                type_o=type_out[j]
                
                o_n = o.cpu().detach().numpy()
                c_o_n = c_o.cpu().detach().numpy()
                ind = np.argpartition(o_n, -10, axis=1)[:, -10:] # top 10 elements (POI-level)
                individual_y_j = individual_y[:, j]#ground truth (location-level)
                
                for k in range(len(individual_y_j)):  
                    if (reset_count[active_users[j]] > 1):
                        continue
                    # resort indices for k:
                    ind_k = ind[k]
                    r = ind_k[np.argsort(-o_n[k, ind_k], axis=0)] # sort top 10 elements descending
                    r = torch.tensor(r)
                    t = individual_y_j[k]
                   
                    
                    # compute MRR:
                    r_kj = o_n[k, :]
                    t_val = r_kj[t]
                    upper = np.where(r_kj > t_val)[0]
                    precision = 1. / (1+len(upper))
                
                    
                    u_iter_cnt[active_users[j]] += 1

                    u_recall1[active_users[j]] += t in r[:1]
                    u_recall5[active_users[j]] += t in r[:5]
                    u_recall10[active_users[j]] += t in r[:10]
                           
                    if t in r[:10] :
                        position=torch.where(r[:10] == t)
                        position=position[0].item()

                        if position == 0:
                            u_mrr1[active_users[j]] += 1/(position+1)
                            u_mrr5[active_users[j]] += 1/(position+1)
                            u_mrr10[active_users[j]] += 1/(position+1)
                        elif position >4:
                            u_mrr10[active_users[j]] += 1/(position+1)      
                        else:
                            u_mrr5[active_users[j]] += 1/(position+1)
                            u_mrr10[active_users[j]] += 1/(position+1)
                    else:
                        continue
                   
                    u_average_precision[active_users[j]] += precision
    
        formatter = "{0:.8f}"
        for j in range(poi_loader.user_count()):
            iter_cnt += u_iter_cnt[j]
            recall1 += u_recall1[j]
            recall5 += u_recall5[j]
            recall10 += u_recall10[j]
            mrr1 += u_mrr1[j]
            mrr5 += u_mrr5[j]
            mrr10 += u_mrr10[j]
            average_precision += u_average_precision[j]
            if (setting.report_user > 0 and (j+1) % setting.report_user == 0):
                print('Report user', j, 'preds:', u_iter_cnt[j], 'recall@1', formatter.format(u_recall1[j]/u_iter_cnt[j]), 'MAP', formatter.format(u_average_precision[j]/u_iter_cnt[j]), sep='\t')
            
        print('recall@1:', formatter.format(recall1/iter_cnt))
        print('recall@5:', formatter.format(recall5/iter_cnt))
        print('recall@10:', formatter.format(recall10/iter_cnt))
        print('mrr@1:', formatter.format(mrr1/iter_cnt))
        print('mrr@5:', formatter.format(mrr5/iter_cnt))
        print('mrr@10:', formatter.format(mrr10/iter_cnt))
        print('MAP', formatter.format(average_precision/iter_cnt))
        print('predictions:', iter_cnt)

        results.append({
            'epoch': e+1,
            'loss_type':loss_type,
            'review':review,
            'poi_weight': poi_weight,
            'cate_weight': cate_weight,
            'type_weight': type_weight,
            'tau': tau,
            'thre': thre,
            'recall@1': recall1/iter_cnt,
            'recall@5': recall5/iter_cnt,
            'recall@10': recall10/iter_cnt,
            'mrr@1': mrr1/iter_cnt,
            'mrr@5': mrr5/iter_cnt,
            'mrr@10': mrr10/iter_cnt,
            'MAP': average_precision/iter_cnt
        })

         
    if results:
        temp_csv = "CAL_current_results.csv"
        fieldnames = results[0].keys()
        with open(temp_csv, mode='w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
    return recall10 / iter_cnt 

complex_samples = []
complex_samples_capacity=1000
threshold_complex = -0.3
revisit_probability = 0.3
tau=setting.tau
mu=setting.mu
thre=setting.thre
poi_weight=setting.poi_weight
cate_weight=setting.cate_weight
type_weight=setting.type_weight
review = setting.review
loss_type = setting.loss_type


# gradient_norm_threshold = 0.1  
# previous_gradient_norm = None

# loss_change_threshold = 2 
# previous_loss = None

# previous_recall10 = float('-inf')  
# current_recall10 = None
# train!
a=1
for e in range(setting.epochs):
    # if (e + 1) % setting.decay_epoch == 0:
    #     tau = mu * tau

    tau = tau * (1 - e / setting.epochs)
    #tau = tau * (e / setting.epochs)
    # x = e / setting.epochs  # Calculate x as the normalized epoch value
    # tau = a ** x  # Update tau based on the a^x function

    h = h0_strategy.on_init(setting.batch_size, setting.device)
    c_h = h0_strategy.on_init(setting.batch_size, setting.device)
    type_h = h0_strategy.on_init(setting.batch_size, setting.device)

    
    for i, (x, c,type, t, s, y,individual_y,c_y,type_y, y_t, y_s, reset_h, c_reset_h,type_reset_h,active_users) in enumerate(dataloader):
        #acquire stage
      
        #for poi
        for j, reset in enumerate(reset_h):
            if reset:
                if setting.is_lstm:
                    hc = h0_strategy.on_reset(active_users[0][j])
                
                    h[0][0, j] = hc[0]
                    h[1][0, j] = hc[1]
                else:
                    h[0, j] = h0_strategy.on_reset(active_users[0][j])
                   
        #for category
        for j, reset in enumerate(c_reset_h):
            if reset:
                if setting.is_lstm:
                    c_hc = h0_strategy.on_reset(active_users[0][j])
                   
                    c_h[0][0, j] = c_hc[0]
                    c_h[1][0, j] = c_hc[1]
                else:
                    c_h[0, j] = h0_strategy.on_reset(active_users[0][j])
        #for type
        for j, reset in enumerate(type_reset_h):
            if reset:
                if setting.is_lstm:
                    type_hc = h0_strategy.on_reset(active_users[0][j])
                   
                    type_h[0][0, j] = type_hc[0]
                    type_h[1][0, j] = type_hc[1]
                else:
                    type_h[0, j] = h0_strategy.on_reset(active_users[0][j])
        
      
        x = x.squeeze().to(setting.device)
        c = c.squeeze().to(setting.device)
        type = type.squeeze().to(setting.device)
        t = t.squeeze().to(setting.device)
        s = s.squeeze().to(setting.device)
        y = y.squeeze().to(setting.device)
        individual_y = individual_y.squeeze().to(setting.device)
        c_y = c_y.squeeze().to(setting.device)
        type_y = type_y.squeeze().to(setting.device)
        y_t = y_t.squeeze().to(setting.device)
        y_s = y_s.squeeze().to(setting.device)                
        active_users = active_users.to(setting.device)       
        optimizer.zero_grad()
       
        loss,cate_loss,type_loss,poi_difficulty,cate_difficulty,type_difficulty, h1, c_h1,type_h1,model_pre,cate_model_pre,type_model_pre = trainer.loss(x, c,type,t, s, y, c_y,type_y,y_t, y_s, h,c_h,type_h, active_users) 
      
        poi_difficulty_norm = (poi_difficulty - poi_difficulty.min()) / (poi_difficulty.max() - poi_difficulty.min())
        cate_difficulty_norm = (cate_difficulty - cate_difficulty.min()) / (cate_difficulty.max() - cate_difficulty.min())
        type_difficulty_norm = (type_difficulty - type_difficulty.min()) / (type_difficulty.max() - type_difficulty.min())


        # adjust weight according to epoch
        initial_weights = [0.5, 1.0, 1.0]  # initial weight
        final_weights = [1, 0.5, 0.5]    # final weight
        poi_weight, cate_weight, type_weight = update_weights(e, setting.epochs, initial_weights, final_weights)



        poi_cate_diff = poi_weight*poi_difficulty_norm+cate_weight*cate_difficulty_norm+type_weight*type_difficulty_norm

       
        difficulty1 = 2 * (poi_cate_diff - poi_cate_diff.min().item()) / (poi_cate_diff.max().item() - poi_cate_diff.min().item()) - 1
        w_j = torch.exp(-tau * (difficulty1 - (torch.mean(difficulty1))) ** 2)
    
    
        gaosi_loss = torch.mean(w_j * loss)

        if setting.loss_type == 'poi_cate':
            loss =loss+cate_loss
        if setting.loss_type == 'ori_poi':
            loss = loss
        if setting.loss_type == 'ori_cate':
            loss = cate_loss
        if setting.loss_type == 'poi_type':
            loss = loss+type_loss
        if setting.loss_type == 'poi_cl':
            difficulty1 = 2 * (poi_difficulty_norm - poi_difficulty_norm.min().item()) / (poi_difficulty_norm.max().item() - poi_difficulty_norm.min().item()) - 1
            w_j = w_j = torch.exp(-tau * (difficulty1 - thre) ** 2)
            gaosi_loss = torch.mean(w_j * loss)
            loss = gaosi_loss
        if setting.loss_type == 'poi_cate_type':
            loss = loss+cate_loss+type_loss
        if setting.loss_type == 'poi_cate_cl':
            loss = gaosi_loss+cate_loss
        
        if setting.loss_type == 'poi_type_cl':
            loss = gaosi_loss+type_loss
        if setting.loss_type == 'poi_cate_type_cl':
            loss = gaosi_loss+cate_loss + type_loss
        if setting.loss_type == 'ori_poi_cate_type_review_exam':
            loss = loss+cate_loss+type_loss
        if setting.loss_type == 'ori_poi_review_exam':
            loss = loss

        # unpdate complex_samples
        if setting.review == '1' and (setting.loss_type == 'poi_cate_type_cl' or setting.loss_type == 'poi_cl' or setting.loss_type == 'poi_cate_cl' or setting.loss_type == 'poi_type_cl' or setting.loss_type == 'ori_poi_cate_type_review_exam' or setting.loss_type == 'ori_poi_review_exam'):
            out, c_out,type_out,_, _ ,_ = trainer.evaluate(x, c,type, t, s, y_t, y_s, h,c_h,type_h, active_users)
            for i in range(x.shape[1]):
                o, c_o,type_o = out[j].cpu().detach().numpy(), c_out[j].cpu().detach().numpy(),type_out[j].cpu().detach().numpy()
                y_j, individual_y_j,c_y_j,type_y_j = y[:, j],individual_y[:, j], c_y[:, j], type_y[:, j]
                for k in range(len(individual_y_j)): 
                    #examine stage
                    ind, c_ind,type_ind = np.argpartition(o[k], -10)[-10:], np.argpartition(c_o[k], -10)[-10:],np.argpartition(type_o[k], -2)[-2:]
                    r, c_r,type_r = ind[np.argsort(-o[k][ind])], c_ind[np.argsort(-c_o[k][c_ind])],type_ind[np.argsort(-type_o[k][type_ind])]
                    if individual_y_j[k].item() not in r or difficulty1[j].item() < thre : 
                        if setting.is_lstm:
                            complex_samples.append((x[:, j], c[:, j], type[:, j], t[:, j], s[:, j], y_t[:, j], y_s[:, j],y[:, j],c_y[:, j],type_y[:, j],active_users.squeeze(0)[j],
                                                (h[0][:, j, :].clone(), h[1][:, j, :].clone()), (c_h[0][:, j, :].clone(), c_h[1][:, j, :].clone()), (type_h[0][:, j, :].clone(), type_h[1][:, j, :].clone()), e,setting.epochs))
                        else:
                            complex_samples.append((x[:, j], c[:, j], type[:, j], t[:, j], s[:, j], y_t[:, j], y_s[:, j],y[:, j],c_y[:, j],type_y[:, j],active_users.squeeze(0)[j],
                                                h[:, j, :].clone(), c_h[:, j, :].clone(),type_h[:, j, :].clone(),e,setting.epochs))

                        if len(complex_samples) > complex_samples_capacity:
                            complex_samples.pop(0)

                        else:
                            continue
                    else:
                        continue
        loss.backward() 
        latest_loss = loss.item()    
        current_gradient_norm = torch.norm(
            torch.cat([p.grad.view(-1) for p in trainer.parameters() if p.grad is not None])
        ).item()    
        
        
        torch.nn.utils.clip_grad_norm_(trainer.parameters(), max_norm=1.0)
        optimizer.step()
      
    scheduler.step()
    previous_gradient_norm = current_gradient_norm
   
    if latest_loss < best_val_loss:
        best_val_loss = latest_loss
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= patience:
        print("Early stopping triggered.")
        break

    if (e+1) % 1 == 0:
        print(f'Epoch: {e+1}/{setting.epochs}')
        print(f'Loss: {latest_loss}')
        print("alpha:",trainer.t_module.lambda_t.data.item(),",beta:",trainer.s_module.lambda_s.data.item())
      
    
    if (e+1) % setting.validate_epoch == 0:
        print('~~~ Test Set Evaluation ~~~')
        current_recall10 = evaluate_test()  
       
    if e+1>15 and setting.review == '1' and len(complex_samples) > 0 and (setting.loss_type == 'poi_cate_type_cl' or setting.loss_type == 'poi_cl' or setting.loss_type == 'poi_cate_cl' or setting.loss_type == 'poi_type_cl' or setting.loss_type == 'ori_poi_cate_type_review_exam' or setting.loss_type == 'ori_poi_review_exam') :
        print("trigger review module")
        review_mode(complex_samples, trainer, optimizer, tau, setting.thre, poi_weight, cate_weight, type_weight)
       

    






