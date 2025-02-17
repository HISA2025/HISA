import torch
from torch.utils.data import Dataset
from enum import Enum
import random
from datetime import datetime

class Split(Enum):
    TRAIN = 0
    TEST = 2
    USE_ALL = 3

class Usage(Enum):
    MIN_SEQ_LENGTH = 0
    MAX_SEQ_LENGTH = 1
    CUSTOM = 2
    

class PoiDataset(Dataset):
    def reset(self):
        # reset training state:
        self.next_user_idx = 0 # current user index to add
        self.active_users = [] # current active users
        self.active_user_seq = [] # current active users sequences
        self.active_user_cate_seq = [] # current active users sequences
        self.user_permutation = [] # shuffle users during training
        self.removed_users=[] # remove users who do not meet the criteria.

        # set active users:
        for i in range(self.user_length):
            self.next_user_idx += 1
            self.active_users.append(i) 
            self.active_user_seq.append(0)
        
        for i in range(len(self.users)):
            self.user_permutation.append(i)

        
    def shuffle_users(self):
        random.shuffle(self.user_permutation)    
        # reset active users:
        self.next_user_idx = 0
        self.active_users = []
        self.active_user_seq = []
      
        for i in range(self.user_length):
            self.next_user_idx += 1
          
            self.active_users.append(self.user_permutation[i]) 
            self.active_user_seq.append(0)
    
    def __init__(self, users, times, coords, locs,individual_locs, cate, type,seq_length, user_length, split, usage, loc_count, custom_seq_count):
        self.users = users
        self.times = times
        self.coords = coords
        self.locs = locs
        self.individual_locs = individual_locs
        self.cate = cate
        self.type = type
        self.labels = []
        self.individual_labels = []
        self.lbl_times = []
        self.lbl_coords = []
        self.cate_labels = []
        self.type_labels = []
        self.sequences = []
        self.individual_sequences = []
        self.cate_sequences = []
        self.type_sequences = []
        self.sequences_times = []
        self.sequences_coords = []
        self.sequences_labels = []
        self.individual_sequences_labels = []
        self.cate_sequences_labels = []
        self.type_sequences_labels = []
        self.sequences_lbl_times = []
        self.sequences_lbl_coords = []
        self.sequences_count = []
        self.Ps = []
        self.Qs = torch.zeros(loc_count, 1)
        self.usage = usage
        self.user_length = user_length
        self.loc_count = loc_count
        self.custom_seq_count = custom_seq_count
     

        self.reset()

        # collect locations:
        for i in range(loc_count):
            self.Qs[i, 0] = i    
        for i, loc in enumerate(locs):
            self.locs[i] = loc[:-1]
            self.labels.append(loc[1:])
            
            # adapt time and coords:
            self.lbl_times.append(self.times[i][1:])
            self.lbl_coords.append(self.coords[i][1:])
            self.times[i] = self.times[i][:-1]
            self.coords[i] = self.coords[i][:-1]

            # adapt intetion:
            self.cate_labels.append(self.cate[i][1:])#
            self.cate[i] = self.cate[i][:-1]

            # adapt type: 
            self.type_labels.append(self.type[i][1:])#
            self.type[i] = self.type[i][:-1]
            # indivudual groundtruth:
            self.individual_labels.append(self.individual_locs[i][1:])#
            self.individual_locs[i] = self.individual_locs[i][:-1]
            



        # split to training / test phase:
        for i, (time, coord, loc,individual_loc,cate,type, label,individual_label,cate_label,type_label, lbl_time, lbl_coord) in enumerate(zip(self.times, self.coords, self.locs,self.individual_locs,self.cate,self.type, self.labels,self.individual_labels,self.cate_labels, self.type_labels,self.lbl_times, self.lbl_coords)):
           
            train_thr = int(len(loc) * 0.8)
            if (split == Split.TRAIN):
                self.times[i] = time[:train_thr]
                self.coords[i] = coord[:train_thr]
                self.locs[i] = loc[:train_thr]
                self.individual_locs[i] = individual_loc[:train_thr]
                self.cate[i] = cate[:train_thr]
                self.type[i] = type[:train_thr]
                self.labels[i] = label[:train_thr]
                self.individual_labels[i] = individual_label[:train_thr]
                self.type_labels[i] = type_label[:train_thr]
                self.lbl_times[i] = lbl_time[:train_thr]
                self.lbl_coords[i] = lbl_coord[:train_thr]
            if (split == Split.TEST):
                self.times[i] = time[train_thr:]
                self.coords[i] = coord[train_thr:]
                self.locs[i] = loc[train_thr:]
                self.individual_locs[i] = individual_loc[train_thr:]
                self.cate[i] = cate[train_thr:]
                self.type[i] = type[train_thr:]
                self.labels[i] = label[train_thr:]
                self.individual_labels[i] = individual_label[train_thr:]
                self.cate_labels[i] = cate_label[train_thr:]
                self.type_labels[i] = type_label[train_thr:]
                self.lbl_times[i] = lbl_time[train_thr:]
                self.lbl_coords[i] = lbl_coord[train_thr:]
            if (split == Split.USE_ALL):
                pass # do nothing
            
        # split location and labels to sequences:
        self.max_seq_count = 0
        self.min_seq_count = 10000000
        self.capacity = 0
        RemoveFlag=True #do not train the squence longger than 120 days 
        for i, (time, coord, loc,individual_loc,cate,type, label, individual_label,cate_label,type_label,lbl_time, lbl_coord) in enumerate(zip(self.times, self.coords, self.locs,self.individual_locs,self.cate, self.type,self.labels, self.individual_labels,self.cate_labels,self.type_labels,self.lbl_times, self.lbl_coords)):
            seq_count = len(loc) // seq_length
            assert seq_count > 0 # fix seq_length and min-checkins in order to have test sequences in a 80/20 split!
            seqs = []
            individual_seqs = []
            cate_seqs = []
            type_seqs = []
            seq_times = []
            seq_coords = []
            seq_lbls = []
            individual_seq_lbls = []
            cate_seq_lbls = []
            type_seq_lbls = []
            seq_lbl_times = []
            seq_lbl_coords = []
            fixed=0 #remove corner case
            for j in range(seq_count):
                start = j * seq_length
                end = (j+1) * seq_length
                # if(RemoveFlag and time[end-1]-time[start]>10368000 and split == Split.TRAIN )
                #     fixed+=1
                #     continue
                seqs.append(loc[start:end])
                individual_seqs.append(individual_loc[start:end])
                cate_seqs.append(cate[start:end])
                type_seqs.append(type[start:end])
                seq_times.append(time[start:end])
                seq_coords.append(coord[start:end])
                seq_lbls.append(label[start:end])
                individual_seq_lbls.append(individual_label[start:end])
                cate_seq_lbls.append(cate_label[start:end])
                type_seq_lbls.append(type_label[start:end])
                seq_lbl_times.append(lbl_time[start:end])
                seq_lbl_coords.append(lbl_coord[start:end])
            seq_count-=fixed
            self.sequences.append(seqs)
            self.individual_sequences.append(individual_seqs)
            self.cate_sequences.append(cate_seqs)
            self.type_sequences.append(type_seqs)
            self.sequences_times.append(seq_times)
            self.sequences_coords.append(seq_coords)            
            self.sequences_labels.append(seq_lbls)
            self.individual_sequences_labels.append(individual_seq_lbls)
            self.cate_sequences_labels.append(cate_seq_lbls)
            self.type_sequences_labels.append(type_seq_lbls)
            self.sequences_lbl_times.append(seq_lbl_times)
            self.sequences_lbl_coords.append(seq_lbl_coords)
            self.sequences_count.append(seq_count)
            self.capacity += seq_count#被过滤后有多少seq
            self.max_seq_count = max(self.max_seq_count, seq_count)#
            self.min_seq_count = min(self.min_seq_count, seq_count)
      
        if (self.usage == Usage.MIN_SEQ_LENGTH):
            print(split,'load', len(users), 'users with min_seq_count', self.min_seq_count, 'batches:', self.__len__())
        if (self.usage == Usage.MAX_SEQ_LENGTH):
            print(split,'load', len(users), 'users with max_seq_count', self.max_seq_count, 'batches:', self.__len__())
        if (self.usage == Usage.CUSTOM):
            print(split,'load', len(users), 'users with custom_seq_count', self.custom_seq_count, 'Batches:', self.__len__())
            
    
    def sequences_by_user(self, idx):
        return self.sequences[idx]
    
    def __len__(self):
        if (self.usage == Usage.MIN_SEQ_LENGTH):
            # min times amount_of_user_batches:
            return self.min_seq_count * (len(self.users) // self.user_length)
        if (self.usage == Usage.MAX_SEQ_LENGTH):
            # estimated capacity:
            estimated = self.capacity // self.user_length
          
            return max(self.max_seq_count, estimated)
        if (self.usage == Usage.CUSTOM):
            return self.custom_seq_count * (len(self.users) // self.user_length)
        raise Exception('Piiiep')
    
    def __getitem__(self, idx):
        ''' Against pytorch convention, we directly build a full batch inside __getitem__.
        Use a batch_size of 1 in your pytorch data loader.
        
        A batch consists of a list of active users,
        their next location sequence with timestamps and coordinates.
        
        y is the target location and y_t, y_s the targets timestamp and coordiantes. Provided for
        possible use.
        
        reset_h is a flag which indicates when a new user has been replacing a previous user in the
        batch. You should reset this users hidden state to initial value h_0.
        '''
        seqs = []
        individual_seqs = []
        times = []
        coords = []
        lbls = []
        individual_lbls = []
        lbl_times = []
        lbl_coords = []
        cate_seqs = []
        cate_lbls = []
        type_seqs = []
        type_lbls = []
      
        reset_h = []
        individual_reset_h = []
        c_reset_h = []
        type_reset_h = []
        for i in range(self.user_length):
            
            i_user = self.active_users[i]
            j = self.active_user_seq[i]

            max_j = self.sequences_count[i_user]
            if (self.usage == Usage.MIN_SEQ_LENGTH):
                max_j = self.min_seq_count
            if (self.usage == Usage.CUSTOM):
                max_j = min(max_j, self.custom_seq_count) # use either the users maxima count or limit by custom count
            if (j >= max_j ):
                # repalce this user in current sequence:
                i_user = self.user_permutation[self.next_user_idx]
                j = 0
                self.active_users[i] = i_user
                self.active_user_seq[i] = j
                
                self.next_user_idx = (self.next_user_idx + 1) % len(self.users)
                while self.user_permutation[self.next_user_idx] in self.active_users:
                    self.next_user_idx = (self.next_user_idx + 1) % len(self.users)
                # TODO: throw exception if wrapped around!
            # use this user:
            reset_h.append(j == 0)
            individual_reset_h.append(j == 0)
            c_reset_h.append(j == 0)
            type_reset_h.append(j == 0)
            seqs.append(torch.tensor(self.sequences[i_user][j]))
            individual_seqs.append(torch.tensor(self.individual_sequences[i_user][j]))
            cate_seqs.append(torch.tensor(self.cate_sequences[i_user][j]))
            type_seqs.append(torch.tensor(self.type_sequences[i_user][j]))
            times.append(torch.tensor(self.sequences_times[i_user][j]))
            coords.append(torch.tensor(self.sequences_coords[i_user][j]))
            lbls.append(torch.tensor(self.sequences_labels[i_user][j]))
            individual_lbls.append(torch.tensor(self.individual_sequences_labels[i_user][j]))
            cate_lbls.append(torch.tensor(self.cate_sequences_labels[i_user][j]))
            type_lbls.append(torch.tensor(self.type_sequences_labels[i_user][j]))
            lbl_times.append(torch.tensor(self.sequences_lbl_times[i_user][j]))
            lbl_coords.append(torch.tensor(self.sequences_lbl_coords[i_user][j]))
            self.active_user_seq[i] += 1

        x = torch.stack(seqs, dim=1)
        individual_x = torch.stack(seqs, dim=1)
        
        c = torch.stack(cate_seqs, dim=1)
        type = torch.stack(type_seqs, dim=1)
        t = torch.stack(times, dim=1)
        s = torch.stack(coords, dim=1)
        y = torch.stack(lbls, dim=1)
        individual_y = torch.stack(individual_lbls, dim=1)
        y_t = torch.stack(lbl_times, dim=1)
        y_s = torch.stack(lbl_coords, dim=1)    
        c_y = torch.stack(cate_lbls, dim=1)
        type_y = torch.stack(type_lbls, dim=1)

              
        return x, c,type,t, s, y,individual_y,c_y,type_y, y_t, y_s, reset_h,c_reset_h,type_reset_h, torch.tensor(self.active_users) 

import pickle
class PoiLoader():
    
    def __init__(self, max_users = 0, min_checkins = 0, seq_length=0):
        self.max_users = max_users
        self.min_checkins = min_checkins
        self.seq_length=seq_length
        self.user2id = {}
        self.poi2id = {}
        self.cate2id = {}
        self.type2id = {}
        self.individual_poi2id= {}
        
        self.users = []
        self.times = []
        self.coords = []
        self.locs = []
        self.individual_locs = []
        self.cate = []
        self.type = []
    
    def poi_dataset(self, seq_length, user_length, split, usage, custom_seq_count = 1):
       
        dataset = PoiDataset(self.users.copy(), self.times.copy(), self.coords.copy(), self.locs.copy(),self.individual_locs.copy(),self.cate.copy(), self.type.copy(),seq_length, user_length, split, usage, len(self.poi2id), custom_seq_count) # crop latest in time
        
        return dataset
    
    def locations(self):
        return len(self.poi2id)
    
    def category(self):
        return len(self.cate2id)
    
    def location_type(self):
        return len(self.type2id)
    
    def individual_locations(self):
        return len(self.individual_poi2id)

    def user_count(self):
        return len(self.users)   
        
    def load(self, file):
      
        self.load_users(file)
        self.load_pois(file)

    
    def load_users(self, file):
        f = open(file, 'r')
        lines = f.readlines()
        prev_user = int(lines[0].split('\t')[0])
        visit_cnt = 0
        check_in_seq=[]
        individual_check_in_seq=[]
        category_check_in_seq=[]
        type_check_in_seq=[]
        for i, line in enumerate(lines):
            tokens = line.strip().split('\t')
           
            user = int(tokens[0])
            location = int(tokens[4])
            category = int(tokens[5])
            type = int(tokens[6])
            individual_location = int(tokens[7])
            
            if user == prev_user:
                visit_cnt += 1
                check_in_seq.append(location)
                category_check_in_seq.append(category)
                type_check_in_seq.append(type)
                individual_check_in_seq.append(individual_location)

            else:

                if visit_cnt >= self.min_checkins:
                    
                    
                    self.user2id[prev_user] = len(self.user2id)
                
                prev_user = user
                check_in_seq=[]
                category_check_in_seq = []
                type_check_in_seq = []
                individual_check_in_seq=[]
                
                visit_cnt = 1
                check_in_seq.append(location)
                individual_check_in_seq.append(individual_location)
                category_check_in_seq.append(category)
                type_check_in_seq.append(type)
                
                if self.max_users > 0 and len(self.user2id) >= self.max_users:
                    break 
   
    def load_pois(self, file):
        f = open(file, 'r')
        lines = f.readlines()
        user_time = []
        user_coord = []
        user_loc = []
        user_cate = []
        user_type = []
        user_individual_loc = []
        
        prev_user = int(lines[0].split('\t')[0])
       
        prev_user = self.user2id.get(prev_user)
       
        for i, line in enumerate(lines):
            
            tokens = line.strip().split('\t')
            user = int(tokens[0])
            if self.user2id.get(user) is None:
                continue # user is not of interrest
           
            user = self.user2id.get(user)
           
            time = (datetime.strptime(tokens[1], "%Y-%m-%dT%H:%M:%SZ") - datetime(1970, 1, 1)).total_seconds() # unix seconds
            lat = float(tokens[2]) # Latitude
            long = float(tokens[3]) # Longitude
            coord = (lat, long)
        
            location = int(tokens[4]) # location 
            category = int(tokens[5]) # intetion 
            type = int(tokens[6]) # type 
            individual_location = int(tokens[7]) # location nr
        
            if self.poi2id.get(location) is None: # get-or-set locations
                self.poi2id[location] = len(self.poi2id)
            location = self.poi2id.get(location)
            if self.individual_poi2id.get(individual_location) is None: # get-or-set locations
                self.individual_poi2id[individual_location] = len(self.individual_poi2id)
            individual_location = self.individual_poi2id.get(individual_location)
           
            if self.cate2id.get(category) is None: # get-or-set category
                self.cate2id[category] = len(self.cate2id)
            category = self.cate2id.get(category)
            if self.type2id.get(type) is None: # get-or-set type
                self.type2id[type] = len(self.type2id)
            type = self.type2id.get(type)
            if user == prev_user:
                user_time.insert(0, time)
                user_coord.insert(0, coord)
                user_loc.insert(0, location)
                user_cate.insert(0, category)
                user_type.insert(0, type)
                user_individual_loc.insert(0, individual_location)
             
            else:
                self.users.append(prev_user)
                self.times.append(user_time)
                self.coords.append(user_coord)
                self.locs.append(user_loc)
                self.individual_locs.append(user_individual_loc)
                self.cate.append(user_cate)
                self.type.append(user_type)
                
                prev_user = user 
                user_time = [time]
                user_coord = [coord]
                user_loc = [location] 
                user_individual_loc = [individual_location] 
                user_cate = [category] 
                user_type = [type] 
                
   
        # process also the latest user in the for loop
        self.users.append(prev_user)
        self.times.append(user_time)
        self.coords.append(user_coord)
        self.locs.append(user_loc)
        self.individual_locs.append(user_individual_loc)
        self.cate.append(user_cate)
        self.type.append(user_type)
        
        with open('CHA_locationReindex.pkl', 'wb') as f:
            pickle.dump(self.poi2id, f)
        with open('CHA_categoryReindex.pkl', 'wb') as f:
            pickle.dump(self.cate2id, f)
        with open('CHA_typeReindex.pkl', 'wb') as f:
            pickle.dump(self.type2id, f)
        with open('CHA_individual_locationReindex.pkl', 'wb') as f:
            pickle.dump(self.individual_poi2id, f)

