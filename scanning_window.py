import numpy as np
import pandas as pd
import neurokit2 as nk
from collections import Counter
from sys import stdin, stdout

def calculate_bpm(signal, sampfreq) -> int:
    """
    Calculate the heart rate in beats per minute (BPM) from an ECG signal.

    Parameters:
    - signal : leadI signal array
    - sampfreq : sampling frequency

    Returns:
    - int: Heart rate in BPM.
    """
    ecg_signal = signal
    sampfreq = sampfreq

    _, rpeaks = nk.ecg_peaks(ecg_signal, sampfreq)

    duration_of_record = len(ecg_signal) / sampfreq

    heart_rate = (len(rpeaks['ECG_R_Peaks']) * 60) / duration_of_record

    return int(heart_rate)

def scan_record(record, window_width, window_step=None):
    
    rhythm_annotation = record._Record__aux

    rhythm_keys = Counter(rhythm_annotation).keys()

    if list(rhythm_keys) == ['']:
        data_within_window = scan_without_interval(record=record,
                                                   window_width=window_width,
                                                   )
    else:
        print(f"There's rhythm annotation. {rhythm_keys} in {record._Record__parent}")
        data_within_window=scan_with_interval(record=record,window_width=window_width)
        

    return data_within_window

def scan_without_interval(record, window_width):
    signal = record._Record__signal
    symbol = record._Record__symbol
    sample = record._Record__sample
    sampfreq = record._Record__sf

    left_end = 0
    right_end = int(window_width * sampfreq)
    heart_rate = calculate_bpm(signal, sampfreq)
    heart_cycle = heart_rate / 60
    types_of_step = 'sec' #input("Choose 'bpm' or 'sec': ")
    no_of_beats_per_step = 1 #int(input("Give number of steps: "))

    if types_of_step == 'bpm':
        window_step = int((no_of_beats_per_step * heart_cycle) * sampfreq)
    elif types_of_step == 'sec':
        window_step = int(window_width * sampfreq)

    ecg_signals = []
    beat_annotations = []
    beat_annotated_points = []
    pac_percentages = []
    pvc_percentages = []
    true_class = []

    while right_end <= len(signal):  
        # print("left_end:", left_end)
        # print("right_end:", right_end)
        # print("window_width:", window_width)
      
        signal_within_window = signal[left_end:right_end]        
        ecg_signals.append(signal_within_window)

        annotated_index = np.intersect1d(np.where(left_end <= sample),
                                            np.where(right_end >= sample))
        symbol_within_window = [symbol[i] for i in annotated_index]
        beat_annotations.append(symbol_within_window)

        sample_within_window = [sample[i] - left_end for i in annotated_index]
        beat_annotated_points.append(sample_within_window)

        beats, count = np.unique(symbol_within_window, return_counts=True)
        segment_beat_annotation_count = dict(zip(beats, count))
        total_count = sum(segment_beat_annotation_count.values())

        pac_percentage = segment_beat_annotation_count.get('A', 0) / total_count * 100
        pac_percentages.append(pac_percentage)
        pvc_percentage = segment_beat_annotation_count.get('V', 0) / total_count * 100
        pvc_percentages.append(pvc_percentage)
        
        true_class.append(determine_true_class(record._Record__label,
                                               pac_percentage,
                                               pvc_percentage))     

              

        left_end += window_step
        right_end = left_end + int(window_width * sampfreq)
        

    parent_record_repeated = [record._Record__parent] * len(ecg_signals)
    label_repeated = [record._Record__label] * len(ecg_signals)
    heart_rate_repeated = [heart_rate] * len(ecg_signals)
    
    signal=pd.DataFrame(ecg_signals)
    info = pd.DataFrame({'beat_annotation_symbols': beat_annotations,
                                       'annotated_samples': beat_annotated_points,
                                       'parent_record': parent_record_repeated,
                                       'pac_percent': pac_percentages,
                                       'pvc_percent': pvc_percentages,
                                       'avg_heart_rate': heart_rate_repeated,
                                       'label': label_repeated,
                                       'true_class': true_class})
    data_within_window=pd.concat([signal,info],axis=1)

    print(f"There are {data_within_window.shape[0]} segments in the record {record._Record__parent}.")

    return data_within_window

def scan_with_interval(record, window_width):
    sampfreq = record._Record__sf
    signal = record._Record__signal
    heart_rate = calculate_bpm(signal, sampfreq)
    heart_cycle = heart_rate / 60
    
    types_of_step = 'bpm'#input("Choose 'bpm' or 'sec': ")
    no_of_beats_per_step = 1 #int(input("Give number of steps: "))
    
    if types_of_step == 'bpm':
        window_step = int((no_of_beats_per_step * heart_cycle) * sampfreq)
    elif types_of_step == 'sec':
        window_step = int(no_of_beats_per_step * sampfreq)
        
    
    nsr_interval = record.get_nsr_interval()
    if nsr_interval:
        nsr_interval=record.get_valid_rhythm_interval(duration=window_width, type='NSR') 
        print(f"NSR interval is from {nsr_interval}")
    
    af_interval = record.get_afib_interval()
    if af_interval:
        af_interval=record.get_valid_rhythm_interval(duration=window_width, type='AF') 
        print(f"AF interval is from {af_interval}")

    if not af_interval and not nsr_interval:
        print ("There is no AF and NSR longer than 30 second segment")
        return pd.DataFrame()
    
    def process_interval(valid_interval,interval_name):
        
        ecg_signals = []
        beat_annotations = []
        beat_annotated_points = []
        pac_percentages = []
        pvc_percentages = []
        true_class = []
        label=interval_name
        # Ensure valid_interval is array-like
        if isinstance(valid_interval, (list, tuple)) and len(valid_interval) > 0:
                      
            for interval in valid_interval:                        
                left_end = interval[0]                
                right_end = left_end+(window_width * sampfreq)            
                
                while right_end <= interval[1]:
                    # Process the interval      
                    signal_within_window = record._Record__signal[left_end:right_end] 
                    
                    annotated_index = np.intersect1d(np.where(left_end <= record._Record__sample),
                                    np.where(right_end >= record._Record__sample))                                    

                    symbol_within_window = [record._Record__symbol[i] for i in annotated_index]                    

                    sample_within_window = [record._Record__sample[i] - left_end for i in annotated_index]
                    
                    # Calculate percentages and true class
                    beats, count = np.unique(symbol_within_window, return_counts=True)
                    segment_beat_annotation_count = dict(zip(beats, count))
                    total_count = sum(segment_beat_annotation_count.values())
                    if total_count:
                        pac_percentage = segment_beat_annotation_count.get('A', 0) / total_count * 100
                        pac_percentages.append(pac_percentage)
                        pvc_percentage = segment_beat_annotation_count.get('V', 0) / total_count * 100
                        pvc_percentages.append(pvc_percentage)
                        
                        true_class.append(determine_true_class(label,
                                                            pac_percentage,
                                                            pvc_percentage)) 
                                 
                        ecg_signals.append(signal_within_window)
                        beat_annotations.append(symbol_within_window)
                        beat_annotated_points.append(sample_within_window)
                        

                    #left_end += int(window_width * sampfreq)     
                    left_end = left_end + window_step               
                    right_end = left_end + int(window_width * sampfreq)
                
        else:
            # Handle case when valid_interval is not array-like
            print("Invalid interval:", valid_interval)
            # Optionally, you can perform alternative actions here
        
        parent_record_repeated = [record._Record__parent] * len(ecg_signals)
        label_repeated = [label] * len(ecg_signals)       
    
        heart_rate_repeated = [heart_rate] * len(ecg_signals)
        signal= pd.DataFrame(ecg_signals)
        info = pd.DataFrame({'parent_record': parent_record_repeated,                                        
                                        'beat_annotation_symbols': beat_annotations,
                                        'annotated_samples': beat_annotated_points,
                                        'pac_percent': pac_percentages,
                                        'pvc_percent': pvc_percentages,
                                        'avg_heart_rate': heart_rate_repeated,
                                        'label': label_repeated,
                                        'true_class': true_class})
        data_within_window=pd.concat([signal,info],axis=1)

        return data_within_window
        
    # Process AF interval
    data_within_af_interval = pd.DataFrame()
    if len(af_interval):
        if record._Record__label:
            data_within_af_interval = process_interval(af_interval,record._Record__label)
        else:
            data_within_af_interval = process_interval(af_interval,'atrial fibrillation')
            

    # Process NSR interval
    data_within_nsr_interval = pd.DataFrame()
    if nsr_interval:
        if record._Record__label:
            data_within_nsr_interval = process_interval(nsr_interval,record._Record__label)
        else:
            data_within_nsr_interval = process_interval(nsr_interval,'non atrial fibrillation')
            


    # Concatenate data from both intervals
    if len(data_within_af_interval) and len(data_within_nsr_interval):
        data_within = pd.concat([data_within_af_interval, data_within_nsr_interval])
        print(f"There are {data_within.shape[0]} segments in the record.")
    elif len(data_within_af_interval) or len(data_within_nsr_interval):
        if len(data_within_af_interval):
            data_within=data_within_af_interval
            print(f"There are {data_within.shape[0]} segments in the record.")
        if len(data_within_nsr_interval):
            data_within=data_within_nsr_interval
            print(f"There are {data_within.shape[0]} segments in the record.")
            
    return data_within


def determine_true_class(label, pac_percentage, pvc_percentage):
    if is_NSR(label, pac_percentage, pvc_percentage):
        if is_pure_NSR(label, pac_percentage, pvc_percentage):
            return 'Pure_NSR'
        else:
            return 'NSR'    
    elif is_PAC(label, pac_percentage, pvc_percentage):
        return 'PAC'
    elif is_PVC(label, pac_percentage, pvc_percentage):
        return 'PVC'
    elif is_AF(label, pac_percentage, pvc_percentage):
        return 'AF'
    else:
        return 'Others'
    
def is_AF(label, pac_percentage, pvc_percentage):
    return label != 'non atrial fibrillation' and pac_percentage == 0 and pvc_percentage == 0

def is_NSR(label, pac_percentage, pvc_percentage):
    return label == 'non atrial fibrillation' and pac_percentage <20 and pvc_percentage <20

def is_PAC(label, pac_percentage, pvc_percentage):
    return label == 'non atrial fibrillation' and pac_percentage >= 20 and pvc_percentage == 0

def is_PVC(label, pac_percentage, pvc_percentage):
    return label == 'non atrial fibrillation' and pac_percentage == 0 and pvc_percentage >= 20

def is_pure_NSR(label, pac_percentage, pvc_percentage):
    return label == 'non atrial fibrillation' and pac_percentage==0 and pvc_percentage==0

        
        
    
    