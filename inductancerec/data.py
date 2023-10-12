# This cell includes the major classes used in our classification analyses
# import matplotlib.pyplot as plt # needed for plotting
import numpy as np # numpy is primary library for numeric array (and matrix) handling
# import scipy as sp
# from scipy import signal
import random
import os
import inductancerec.utility

class SensorData:
    '''
    Contains the inductance data as numpy arrays
    '''
     
    def __init__(self, sensor_type, data_id, time, sensor_time, ind_data, ind_raw_data):
        '''
        All arguments are numpy arrays except sensor_type, which is a str
        '''
        self.sensor_type = sensor_type
        
        # On my mac, I could cast as straight-up int but on Windows, this failed
        # This is because on Windows, a long is 32 bit but on Unix, a long is 64bit
        # So, forcing to int64 to be safe. See: https://stackoverflow.com/q/38314118
        self.time = time.astype(np.float64) # timestamps are in milliseconds
        
        # sensor_time comes from the evaluation board. it's in milliseconds 
        # which returns the number of milliseconds passed since the board began running the current program.
        self.sensor_time = sensor_time.astype(np.float64) # timestamps are in milliseconds
        
        self.ind_data = ind_data.astype(float)
        
        # Create placeholders for processed data
        self.ind_data_p = None

        length_in_milli = 0
        for i, val in enumerate(self.time):
            if i != 0:
                length_in_milli = length_in_milli + val
            
        # self.length_in_secs = (self.time[-1] - self.time[0]) / 1000.0
        self.length_in_secs = length_in_milli / 1000.0
        self.sampling_rate = len(self.time) / self.length_in_secs 
        
    def length(self):
        '''
        Returns length (in rows). Note that all primary data structures: time and inductance data
        are the same length. So just returns len(self.ind_data). Depending on the preprocessing alg,
        the processed data may be a different length than unprocessed
        '''
        return len(self.ind_data)
        
    def get_data(self):
        '''
        Returns a dict of numpy arrays for each data, e.g., inductance.
        '''
        return {"inductance":self.ind_data}
    
    def get_processed_data(self):
        '''
        Returns a dict of numpy arrays for each data, e.g., inductance.
        '''
        return {"ind_data_p":self.ind_data_p}

    def __str__(self):
        return "{}: {} samples {:.2f} secs {:.2f} Hz".format(self.sensor_type, self.length(),
                                                    self.length_in_secs, self.sampling_rate)


class Trial:
    '''
    A trial is one deformation recording and includes an inductance Sensor SensorData object
    '''
    
    def __init__(self, deformation_name, trial_num, log_filename_with_path):
        '''
        We actually parse the sensor log files in the constructor--this is probably bad practice
        But offers a relatively clean solution
        
        deformation_name : the deformation name (as a str)
        trial_num : the trial number (collect 5 or maybe 10 trials per deformation)
        log_filename_with_path : the full path to the filename (as a str)
        '''
        self.deformation_name = deformation_name
        self.trial_num = trial_num
        self.log_filename_with_path = log_filename_with_path
        self.log_filename = os.path.basename(log_filename_with_path)
        
        # unpack=True puts each column in its own array, see https://stackoverflow.com/a/20245874
        # I had to force all types to strings because auto-type inferencing failed
        parsed_ind_log_data = np.genfromtxt(log_filename_with_path, delimiter=',', 
                              dtype=str, encoding=None, skip_header=1, unpack=True)
        
        # The asterisk is really cool in Python. It allows us to "unpack" this variable
        # into arguments needed for the SensorData constructor. Google for "tuple unpacking"
        
        self.inductance = SensorData("Spring-25-50-20-5", *parsed_ind_log_data)
    
    def get_ground_truth_deformation_name(self):
        '''Returns self.deformation_name'''
        return self.deformation_name
        
    def length(self):
        '''Gets the length of the trial in samples'''
        return len(self.inductance.ind_data)
    
    def get_start_time(self):
        '''Gets the start timestamp'''
        return self.inductance.time[0]
    
    def get_end_time(self):
        '''Gets the end timestamp'''
        return self.inductance.time[-1]
    
    def get_end_time_as_string(self):
        '''Utility function that returns the end time as a nice string'''
        return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.get_end_time() / 1000))
    
    def __str__(self):
         return "'{}' : Trial {} from {}".format(self.deformation_name, self.trial_num, self.log_filename)
        
class DeformationSet:
    '''
    Container for a single set of deformation and trials
    '''

    DEFAULT_DEFORMATION_NAMES = set(['Compressing', "Twisting", "Bending", "Extending", "Calibration"])
    
    DEFORMATION_NAMES_WITHOUT_CUSTOM = None
    
    def __init__(self, deformation_log_path):
        '''
        After calling the constructor, you must call *load* and then *preprocess*
        
        Parameters:
        deformation_log_path: path to the deformation log dir
        '''

        self.path = deformation_log_path
        self.name = self.get_base_path() # do not change the name, it's used as an dict key

        self.DEFORMATION_NAMES_WITHOUT_CUSTOM = set(self.DEFAULT_DEFORMATION_NAMES)

    def load(self):
        '''Loads the deformation trials.'''
        
        # Our primary object that maps a deformation name to a list of Trial objects
        self.map_deformations_to_trials = self.__parse_deformation_trials(self.path)   
    
    def __parse_deformation_trials(self, path_to_dir):
        '''
        Parses and creates Trial objects for all csv files in the given dir. 
        It's not necessary that you understand this code
        
        Parameters:
        path_to_dir: the path to the deformation logs
        
        Returns:
        dict: a dict() mapping (str: deformation_name) to (list: Trial objects)
        '''
        csv_filenames = inductancerec.utility.find_csv_filenames(path_to_dir)

        print("Found {} csv files in {}".format(len(csv_filenames), path_to_dir))

        map_deformation_name_to_trial_list = dict()
        map_deformation_name_to_map_num_to_map_sensor_to_file = dict() # use this to correctly order trials
        for csvFilename in csv_filenames:

            # parse filename into meaningful parts
            # print(csvFilename)
            filename_no_ext = os.path.splitext(csvFilename)[0];

            deformation_name = None
            sensor_name = "Spring-25-50-20-5" # currently only one sensor but could expand to more

            # Added this conditional because Windows machines created differently formatted
            # filenames from Macs. Windows machines automatically replaced the character "'"
            # with "_", which affects filenames like "'Z'_1556730840228_206.csv"
            # which come out like "_Z__1557937136974_211.csv" instead
            filename_parts = filename_no_ext.split("_")
            deformation_name = filename_parts[0]
            No = filename_parts[1]
            print("deformation_name={} No.{}".format(deformation_name, No))

            if deformation_name not in map_deformation_name_to_map_num_to_map_sensor_to_file:
                map_deformation_name_to_map_num_to_map_sensor_to_file[deformation_name] = dict()

            if No not in map_deformation_name_to_map_num_to_map_sensor_to_file[deformation_name]:
                map_deformation_name_to_map_num_to_map_sensor_to_file[deformation_name][No] = dict()

            map_deformation_name_to_map_num_to_map_sensor_to_file[deformation_name][No][sensor_name] = csvFilename
            # print (map_deformation_name_to_map_endtime_to_map_sensor_to_file)

        print("Found {} deformations".format(len(map_deformation_name_to_map_num_to_map_sensor_to_file)))

        # track the longest array
        max_array_length = -1
        trial_with_most_sensor_events = None

        # Now we need to loop through the data and sort each deformation set by timems values 
        # (so that we have trial 1, 2, 3, etc. in order)
        for deformation_name, map_num_to_map_sensor_to_file in map_deformation_name_to_map_num_to_map_sensor_to_file.items():
            
            print(deformation_name)
            print(map_num_to_map_sensor_to_file)
            
            deformation_trial_num = 0
            map_deformation_name_to_trial_list[deformation_name] = list()
            for No in sorted(map_num_to_map_sensor_to_file.keys()):
                map_sensor_to_file = map_num_to_map_sensor_to_file[No]

                log_filename_with_path = os.path.join(path_to_dir, map_sensor_to_file["Spring-25-50-20-5"])
                deformation_trial = Trial(deformation_name, deformation_trial_num, log_filename_with_path)
                map_deformation_name_to_trial_list[deformation_name].append(deformation_trial)

                if max_array_length < len(deformation_trial.inductance.ind_data):
                    max_array_length = len(deformation_trial.inductance.ind_data)
                    trial_with_most_sensor_events = deformation_trial

                deformation_trial_num = deformation_trial_num + 1

            print("Found {} trials for '{}'".format(len(map_deformation_name_to_trial_list[deformation_name]), deformation_name))

        # Print out some basic information about our logs
        print("Max trial length across all deformations is '{}' Trial {} with {} sensor events.".
              format(trial_with_most_sensor_events.deformation_name, trial_with_most_sensor_events.trial_num, max_array_length))
        list_samples_per_second = list()
        list_total_sample_time = list()
        for deformation_name, trial_list in map_deformation_name_to_trial_list.items():
            for trial in trial_list: 
                list_samples_per_second.append(trial.inductance.sampling_rate)
                list_total_sample_time.append(trial.inductance.length_in_secs)

        print("Avg samples/sec across {} sensor files: {:0.1f}".format(len(list_samples_per_second), 
                                                                       sum(list_samples_per_second)/len(list_samples_per_second)))
        print("Avg sample length across {} sensor files: {:0.1f}s".format(len(list_total_sample_time), 
                                                                          sum(list_total_sample_time)/len(list_total_sample_time)))
        print()

        return map_deformation_name_to_trial_list
    
    def get_trials(self, deformation_name):
        '''Returns a list of trials for this deformation name sorted chronologically'''
        return self.map_deformations_to_trials[deformation_name]
    
    def get_all_trials(self):
        '''Gets all trials sorted chronologically'''
        trials = list()
        for deformation_name, trial_list in self.map_deformations_to_trials.items():
            trials += trial_list
            
        trials.sort(key=lambda x: x.get_start_time())
        return trials
    
    def get_all_trials_except(self, trial):
        '''Gets all the trials except the given trial'''
        trials = self.get_all_trials()
        trials.remove(trial)
        return trials     
    
    def get_trials_that_overlap(self, start_timestamp, end_timestamp):
        '''Returns the trials that overlap the start and end timestamps (inclusive)'''
        matched_trials = list()
        trials = self.get_all_trials()
        for trial in trials:
            if trial.get_end_time() >= start_timestamp and trial.get_start_time() <= end_timestamp:
                matched_trials.append(trial)
            elif trial.get_start_time() > end_timestamp:
                break # trials are ordered, no need to continue through list
        return matched_trials
    
    def get_longest_trial(self):
        '''Returns the longest trial (based on num rows recorded)'''
        longest_trial_length = -1
        longest_trial = None
        for deformation_name, trial_list in self.map_deformations_to_trials.items():
            for trial in trial_list:
                if longest_trial_length < len(trial.inductance.ind_data):
                    longest_trial_length = len(trial.inductance.ind_data)
                    longest_trial = trial
        return longest_trial
    
    def get_base_path(self):
        '''Returns the base path of self.path'''
        return os.path.basename(os.path.normpath(self.path))
    
    def get_num_deformations(self):
        '''Returns the number of deformations'''
        return len(self.map_deformations_to_trials)
    
    def get_trials_for_deformation(self, deformation_name):
        '''Returns trials for the given deformation name'''
        return self.map_deformations_to_trials[deformation_name]
        
    def get_min_num_of_trials(self):
        '''
        Returns the minimum number of trials across all deformations (just in case we accidentally recorded a 
        different number. We should have the same number of trials across all deformations)
        '''
        min_num_trials = -1 
        for deformation_name, trials in self.map_deformations_to_trials.items():
            if min_num_trials == -1 or min_num_trials > len(trials):
                min_num_trials = len(trials)
        return min_num_trials

    def get_total_num_of_trials(self):
        '''Returns the total number of trials'''
        numTrials = 0 
        for deformation_name, trialSet in self.map_deformations_to_trials.items():
            numTrials = numTrials + len(trialSet)
        return numTrials
    
    def get_random_deformation_name(self):
        '''Returns a random deformation name from within this DeformationSet'''
        deformation_names = list(self.map_deformations_to_trials.keys())
        rand_deformation_name = deformation_names[random.randint(0, len(deformation_names) - 1)]
        return rand_deformation_name
    
    def get_random_trial_for_deformation(self, deformation_name):
        '''Returns a random trial for the given deformation name'''
        trials_for_deformation = self.map_deformations_to_trials[deformation_name]
        return trials_for_deformation[random.randint(0, len(trials_for_deformation) - 1)]
    
    def get_random_trial(self):
        '''Returns a random trial'''
        rand_deformation_name = self.get_random_deformation_name()
        print("rand_deformation_name", rand_deformation_name)
        trials_for_deformation = self.map_deformations_to_trials[rand_deformation_name]
        return trials_for_deformation[random.randint(0, len(trials_for_deformation) - 1)]
    
    def get_deformation_names_sorted(self):
        '''Returns a sorted list of deformation names'''
        return sorted(self.map_deformations_to_trials.keys())

    def get_deformation_names_filtered(self, filter_names):
        '''Returns the deformation names except for those in the filter_names list'''
        filter_names = set(filter_names)
        deformation_names = list()
        for deformation_name in self.map_deformations_to_trials.keys():
            if deformation_name not in filter_names:
                deformation_names.append(deformation_names)
        
        return sorted(deformation_names)
    
    def __str__(self):
         return "'{}' : {} deformations and {} total trials".format(self.path, self.get_num_deformations(), 
                                                                self.get_total_num_of_trials())

# Deformation set utility functions

# Gets a random deformation set
def get_random_deformation_set(map_deformation_sets):
    '''
    Returns a random deformation set
    '''
    deformation_set_names = list(map_deformation_sets.keys())
    rand_deformation_set_name = deformation_set_names[random.randint(0, len(deformation_set_names) - 1)]
    return map_deformation_sets[rand_deformation_set_name]

def get_deformation_set(map_deformation_sets, key):
    '''
    Gets the deformation set for the given key
    '''
    return map_deformation_sets[key]

def get_deformation_set_with_str(map_deformation_sets, s):
    '''
    Gets the deformation set containing the str s 
    '''
    for deformation_set_name, deformation_set in map_deformation_sets.items():
        if s in deformation_set_name:
            return deformation_set
    
    print(f"We could not find '{s}' in map_deformation_sets")

    return None

def get_deformation_sets_with_str(map_deformation_sets, s):
    '''
    Gets all deformation sets with s in the name
    
    s: can be a string or a collection of strings
    '''
    deformation_sets = []
    for base_path, deformation_set in map_deformation_sets.items():
        if isinstance(s, str):
            if s in base_path:
                deformation_sets.append(deformation_set)
        else:
            for i_str in s:
                if i_str in base_path:
                    deformation_sets.append(deformation_set)

    if len(deformation_sets) <= 0:
        print(f"We found no deformation sets with the string '{s}'")
    
    return deformation_sets

def get_random_deformation_set(map_deformation_sets):
    '''
    Returns a random deformation set
    '''
    import random
    keys = list(map_deformation_sets.keys())
    rand_key = random.choice(keys)
    rand_deformation_set = map_deformation_sets[rand_key]
    return rand_deformation_set

def get_deformation_set_names_sorted(map_deformation_sets):
    '''
    Returns a list of deformation set names sorted by name
    '''
    return sorted(list(map_deformation_sets.keys()))

def get_all_deformation_sets(map_deformation_sets):
    '''
    Gets all of the deformation sets
    '''
    return map_deformation_sets.values()

def get_all_deformation_sets_except(map_deformation_sets, filter):
    '''
    Gets all of the deformation sets except filter. Filter can be a string
    or a list of strings
    '''
    if isinstance(filter, str):
        filter = [filter]
    
    deformation_sets = []
    for deformation_set_name, deformation_set in map_deformation_sets.items():
        if filter.count(deformation_set_name) <= 0:
            deformation_sets.append(deformation_set)

    return deformation_sets