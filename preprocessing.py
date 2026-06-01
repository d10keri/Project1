import os
import torch
from scipy.signal import butter, filtfilt, stft, resample
import numpy as np
import matplotlib.pyplot as plt
import pyedflib
from configs.autocfg import cfg
import pywt  
import wfdb



class Filter:
    def __init__(self, segment_duration=4, nperseg=32, lowcut=10, highcut=100, target_fs=250):
        self.segment_duration = segment_duration
        self.nperseg = nperseg
        self.lowcut = lowcut
        self.highcut = highcut
        self.target_fs = target_fs

    @staticmethod
    def normalize_signal(signal):
        return (signal - np.min(signal)) / (np.max(signal) - np.min(signal)) * 2 - 1

    @staticmethod
    def bandpass_filter(data, lowcut, highcut, fs, order=4):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data)

    def resample_signal(self, signal, original_fs):
        num_samples = int(len(signal) * self.target_fs / original_fs)
        return resample(signal, num_samples)

    def segment_signal(self, signal, fs):
        segment_length = int(self.segment_duration * fs)
        return [signal[i:i+segment_length] for i in range(0, len(signal), segment_length) if len(signal[i:i+segment_length]) == segment_length]


class CWTPreprocessor:
    # Định nghĩa các phương thức trước __init__

    def scale_range_for_band(self, f_min, f_max, wavelet, sampling_rate, num=17):
        """
        Trả về mảng scale tương ứng với dải tần số [f_min, f_max] (Hz).
        """
        freqs = np.linspace(f_min, f_max, num)
        scales = self.frequency_to_scale(freqs, wavelet, sampling_rate)
        return scales, freqs

    def frequency_to_scale(self, frequencies, wavelet, sampling_rate):
        """
        Chuyển đổi một mảng tần số (Hz) sang scale, sử dụng hàm scale2frequency của PyWavelets.
        """
        frequencies = np.asarray(frequencies)
        # PyWavelets trả về tần số chuẩn hóa: freq = pywt.scale2frequency(wavelet, scale)
        # => scale = pywt.scale2frequency(wavelet, 1.0) * sampling_rate / frequency
        scale_1 = pywt.scale2frequency(wavelet, 1.0)
        scales = scale_1 * sampling_rate / frequencies
        return scales

    # Khởi tạo đối tượng
    def __init__(self, data_dir, output_dir, segment_duration=4, lowcut=10, highcut=100, target_fs=250, wavelet='mexh', f_min=10, f_max=100, num_freqs=10):
        # Lưu các thông số vào các thuộc tính của đối tượng
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.segment_duration = segment_duration
        self.lowcut = lowcut
        self.highcut = highcut
        self.target_fs = target_fs
        self.wavelet = wavelet

        # Tính toán scales và freqs từ tần số min/max và wavelet
        self.scales, self.freqs = self.scale_range_for_band(f_min, f_max, wavelet, target_fs, num_freqs)
        
        # Tạo thư mục output nếu chưa có
        os.makedirs(self.output_dir, exist_ok=True)

    @staticmethod
    def normalize_signal(signal):
        return (signal - np.min(signal)) / (np.max(signal) - np.min(signal)) * 2 - 1
    
    @staticmethod
    def bandpass_filter(data, lowcut, highcut, fs, order=4):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data)


 
    def resample_signal(self, signal, original_fs):
        num_samples = int(len(signal) * self.target_fs / original_fs)
        return resample(signal, num_samples)

    def segment_signal(self, signal, fs):
        segment_length = int(self.segment_duration * fs)
        segments = [(signal[i:i+segment_length], (i, i+segment_length))
                    for i in range(0, len(signal), segment_length) 
                    if len(signal[i:i+segment_length]) == segment_length]
        return segments
    
    def apply_cwt(self, signal):
        """
        Áp dụng Continuous Wavelet Transform (CWT) lên tín hiệu sử dụng wavelet và scales đã được tính toán.
        """
        if np.all(signal == 0) or len(signal) < 10:  # Kiểm tra nếu tín hiệu không có giá trị
            print(f"Warning: Empty or too short signal detected. Skipping CWT computation.")
            return np.zeros((len(self.scales), len(signal))), np.array(self.scales)  # Trả về ma trận 0 thay vì lỗi
        
        # Sử dụng self.wavelet và self.scales đã được lưu trong đối tượng
        cwtmatr, freqs = pywt.cwt(signal, self.scales, self.wavelet, 1.0 / self.target_fs)
        
        return cwtmatr, freqs
    
    @staticmethod
    def normalize_cwt(cwt_values):
        cwt_min = np.min(cwt_values)
        cwt_max = np.max(cwt_values)
        normalized_cwt = (cwt_values - cwt_min) / (cwt_max - cwt_min)
        return normalized_cwt, cwt_min, cwt_max

    # def process_single_file(self, edf_file_path):
    #     edf_reader = pyedflib.EdfReader(edf_file_path)
    #     object_name = os.path.splitext(os.path.basename(edf_file_path))[0]  # Tên file không bao gồm đuôi

    #     # Số lượng kênh và nhãn kênh
    #     num_channels = edf_reader.signals_in_file
    #     channel_labels = edf_reader.getSignalLabels()

    #     # Đọc tín hiệu và lấy tần số lấy mẫu ban đầu
    #     signals = [edf_reader.readSignal(i) for i in range(num_channels)]
    #     original_sample_rates = edf_reader.getSampleFrequencies()
    #     edf_reader.close()

    #     # Chuẩn hóa, lấy mẫu lại và lọc tín hiệu
    #     resampled_signals = [
    #         self.resample_signal(self.normalize_signal(signal), fs)
    #         for signal, fs in zip(signals, original_sample_rates)
    #     ]
    #     filtered_signals = [
    #         self.bandpass_filter(signal, self.lowcut, self.highcut, self.target_fs)
    #         for signal in resampled_signals
    #     ]
    def process_single_file(self, record_path):
    # record_path: đường dẫn đến file .hea (vd: "/path/to/r01")
        record = wfdb.rdrecord(record_path)
        object_name = os.path.basename(record_path)

        signals = record.p_signal.T  # Kích thước: (num_channels, num_samples)
        original_sample_rates = [record.fs] * signals.shape[0]  # Tần số lấy mẫu chung cho tất cả kênh
        num_channels = signals.shape[0]
        channel_labels = record.sig_name
        print(f"File {object_name}: channels = {channel_labels}")

    # Chuẩn hóa, lấy mẫu lại và lọc tín hiệu
        resampled_signals = [
            self.resample_signal(self.normalize_signal(signals[ch]), original_sample_rates[ch])
            for ch in range(num_channels)
        ]
        filtered_signals = [
            self.bandpass_filter(signal, self.lowcut, self.highcut, self.target_fs)
            for signal in resampled_signals
        ]


        # Tính CWT và chuẩn hóa kết quả theo từng kênh
        results = []
        for channel_index, (channel_label, signal) in enumerate(zip(channel_labels, filtered_signals)):
            # segments = self.segment_signal(signal, self.target_fs) #(1)
            #(2)
            segments = self.segment_signal(signal, self.target_fs)#(2)

            cwt_results_norm = []
            for segment, idx in segments:
                cwtmatr, freqs = self.apply_cwt(segment)
                # cwt_coeffs, frequencies = self.apply_cwt(segment)#(1)
                # gỡ các dấu cmt vs nhau theo cặp ra để test


                real = np.real(cwtmatr)
                
                real_norm, real_min, real_max = self.normalize_cwt(real)
                
                cwt_results_norm.append({
                    "object_name": object_name,
                    "channel_label": channel_label,
                    "frequencies": torch.tensor(freqs, dtype=torch.float32),
                    "segment": segment,
                    "chunk_idx": idx,
                    "real_norm": torch.tensor(real_norm, dtype=torch.float32),
                    
                    
                    "real_min": real_min,
                    "real_max": real_max,
                    
                    
                })

            results.append({
                "object_name": object_name,
                "channel_label": channel_label,
                "cwt_results": cwt_results_norm,
                "segments": segments
            })
        return results

    def process_all_files(self):
        # edf_files = [f for f in os.listdir(self.data_dir) if f.endswith('.edf')]

        # for edf_file in edf_files:
        #     edf_file_path = os.path.join(self.data_dir, edf_file)
        #     results = self.process_single_file(edf_file_path)
        hea_files = [f[:-4] for f in os.listdir(self.data_dir) if f.endswith('.hea')]  # lấy tên file không đuôi .hea

        for record_name in hea_files:
            record_path = os.path.join(self.data_dir, record_name)
            results = self.process_single_file(record_path)
            for result in results:
                output_path = os.path.join(self.output_dir, f"{result['object_name']}_{result['channel_label']}_cwt_norm.pt")
                torch.save(result["cwt_results"], output_path)
                print(f"Saved CWT results for {result['object_name']}, channel '{result['channel_label']}' to {output_path}")

        print("Processing completed for all files!")


    def prepare_training_data(self, output_path):
        os.makedirs(output_path, exist_ok=True)
        # Define input channels and target channel
        input_channels = ['AECG1', 'AECG2', 'AECG3', 'AECG4']
        # target_channel = 'Direct_1'

        # Prepare training data
        inputs = []
        # targets = []
        raw_input_segments = []
        # raw_target_segments = []
        input_channel_labels = []
        chunk_indies = []
        object_names = []

        # # Debug input shape
        # print(f"Input shape: {inputs[0].shape}")  # Kiểm tra số kênh đầu vào
        # print(f"Target shape: {targets[0].shape}")  # Kiểm tra số kênh đầu ra


        cwt_files = [f for f in os.listdir(self.output_dir) if f.endswith('.pt')]

        # Group files by object names
        object_files = {}
        for cwt_file in cwt_files:
            object_name, channel_label = cwt_file.split('_', 1)
            channel_label = channel_label.replace('_cwt_norm.pt', '')
            if object_name not in object_files:
                object_files[object_name] = {}
            object_files[object_name][channel_label] = cwt_file

        # Create input-output pairs
        # for object_name, channels in object_files.items():
        #     if target_channel not in channels:
        #         print(f"Warning: Target channel '{target_channel}' missing for object '{object_name}'")
        #         continue

        #     target_data = torch.load(os.path.join(self.output_dir, channels[target_channel]), weights_only=False)

        #     for input_channel in input_channels:
        #         if input_channel not in channels:
        #             print(f"Warning: Input channel '{input_channel}' missing for object '{object_name}'")
        #             continue

        #         input_data = torch.load(os.path.join(self.output_dir, channels[input_channel]))

        #         # Iterate through the data to create input-output pairs
        #         # num_samples = len(input_data['real_norm'])
        #         num_segments = len(input_data)

        #         for i in range(num_segments):
        #          real_input = input_data[i]['real_norm'].numpy()
        #          raw_input_segment = input_data[i]['segment']
        #          idx = input_data[i]['chunk_idx']
        #          object_name = input_data[i]['object_name']
        #          input_channel_label = input_data[i]['channel_label']
        #          input_image = real_input  # Giữ lại real_input thay vì kết hợp với imag_input

        #          real_target = target_data[i]['real_norm'].numpy()
        #          raw_target_segment = input_data[i]['segment']
        #          target_image = real_target  # Giữ lại real_target thay vì kết hợp với imag_target

        #          inputs.append(torch.tensor(input_image, dtype=torch.float32))
        #          targets.append(torch.tensor(target_image, dtype=torch.float32))
        #          raw_input_segments.append(raw_input_segment)
        #          raw_target_segments.append(raw_target_segment)
        #          input_channel_labels.append(input_channel_label)
        #          chunk_indies.append(idx)
        #          object_names.append(object_name)
        for object_name, channels in object_files.items():
            # Kiểm tra đủ 4 kênh
            if not all(ch in channels for ch in input_channels):
                print(f"Warning: Missing input channels for {object_name}, skipping...")
                continue

# Lấy chỉ kênh đầu tiên làm input
            first_channel = input_channels[0]
            path = os.path.join(self.output_dir, channels[first_channel])
            ch_data = torch.load(path)

            num_segments = len(ch_data)

            for i in range(num_segments):
                real_input = ch_data[i]['real_norm'].numpy()
                input_image = real_input

                inputs.append(torch.tensor(input_image, dtype=torch.float32))
                raw_input_segments.append(ch_data[i]['segment'])
                input_channel_labels.append([first_channel])
                chunk_indies.append(ch_data[i]['chunk_idx'])
                object_names.append(object_name)

        # Save the prepared dataset as a dictionary
        training_dataset = {
            "inputs": inputs,
            "raw_input_segments": raw_input_segments,
            "input_channel_labels": input_channel_labels,
            "object_names": object_names,
            "chunk_indies": chunk_indies
        }
        torch.save(training_dataset, os.path.join(output_path, cfg.get("save_dataset_name")))
        print(f"Training dataset saved to {os.path.join(output_path, cfg.get('save_dataset_name'))}")

    def visualize_combined(self, signal, fs, channel_label, file_path, chunk_number=0):
        segments = self.segment_signal(signal, fs)
        if chunk_number >= len(segments):
            raise ValueError(f"Chunk number {chunk_number} is out of range.")

        segment = segments[chunk_number]
        freqs, cwtmatr = self.apply_cwt(segment)
        
        cwt_data = torch.load(file_path)
        


        stored_freqs = cwt_data['frequencies'][chunk_number].numpy()
        real_part = cwt_data['real'][chunk_number].numpy()
     
        
        time_axis = np.linspace(0, self.segment_duration, len(segment))

        fig, axes = plt.subplots(4, 1, figsize=(10, 12), constrained_layout=True)
        

        axes[0].plot(time_axis, segment)
        axes[0].set_title(f"Original Signal (Chunk {chunk_number})")
        axes[0].set_ylabel("Amplitude")
        axes[0].set_xlabel("Time (s)")
        axes[0].set_xlim(0, self.segment_duration)
        axes[0].set_ylim(segment.min(), segment.max())  # Đặt gốc 0
        axes[0].spines['top'].set_visible(False)
        axes[0].spines['right'].set_visible(False)
        
        im1 = axes[1].pcolormesh(np.arange(cwtmatr.shape[1]), freqs, np.abs(cwtmatr), shading='auto', cmap='hot')
        axes[1].set_title(f"CWT Magnitude (Chunk {chunk_number})")
        axes[1].set_xlabel("Time (Index)")
        axes[1].set_ylabel("Frequency (Hz)")
        fig.colorbar(im1, ax=axes[1])
        
        im2 = axes[2].pcolormesh(np.arange(real_part.shape[1]), freqs, np.abs(real_part), shading='auto', cmap='hot')
        axes[2].set_title(f"Stored CWT Real Magnitude (Chunk {chunk_number})")
        axes[2].set_xlabel("Time (Index)")
        axes[2].set_ylabel("Frequency (Hz)")
        fig.colorbar(im2, ax=axes[2])
        

        plt.show()

        
# Example usage
if __name__ == "__main__":
    data_dir = "/home/bmestaging/nqthinh/FECG_extraction_STFT_sonng/set-a/"
    # edf_file_path = "/home/bmestaging/nqthinh/FECG_extraction_STFT_sonng/abdominal-and-direct-fetal-ecg-database-1.0.0/r01.edf"
    output_dir = "cwt_results_pcdb"


    preprocessor = CWTPreprocessor(data_dir, output_dir, target_fs=250)
    preprocessor.process_all_files()

    # Prepare training dataset
    preprocessor.prepare_training_data("training_dataset_cwt_pcdb")

    # # Visualize combined results
    # example_file_path = os.path.join(output_dir, "r01_Abdomen_1_cwt_norm.pt")

    # edf_reader = pyedflib.EdfReader(edf_file_path)
    # original_signal = edf_reader.readSignal(0)  # Select the first channel
    # fs_original = edf_reader.getSampleFrequency(0)  # Get the sampling frequency of the first channel
    # edf_reader.close()

    # resampled_signal = preprocessor.resample_signal(preprocessor.normalize_signal(original_signal), fs_original)

    # preprocessor.visualize_combined(resampled_signal, preprocessor.target_fs, "Abdomen_1", example_file_path, chunk_number=1)

    


