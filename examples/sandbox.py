# import numpy as np
# import matplotlib.pyplot as plt
# fs = 100.01
# twin = 30

# n_samples = np.round(twin*fs) + 2
# print(n_samples)

# total_length_s = 60*60*6

# tstart_des = np.arange(0, total_length_s, twin)
# tstart_act = np.zeros_like(tstart_des)
# tend_act = np.zeros_like(tstart_des)
# tend_act[0] = (n_samples-1)/fs
# for n in range(1, len(tstart_des)):
#     tstart_act[n] = tend_act[n-1] - 1/fs
#     tend_act[n] = tstart_act[n] + n_samples/fs

# plt.figure()
# plt.subplot(1,2,1)
# plt.plot(tstart_act)
# plt.plot(tend_act)
# plt.plot(tstart_des)
# plt.legend(['actual start time', 'actual end time', 'desired start time'])

# plt.subplot(1,2,2)
# plt.plot(tstart_act - tstart_des)
# plt.plot(-1/fs*np.ones(tstart_des.shape), 'k--')
# plt.plot(+1/fs*np.ones(tstart_des.shape), 'k--')
# plt.show()



import numpy as np
import pandas as pd

class TestLoader:
    def __init__(self, fs, time_window_s=30, chunk_pad_s=0.1):
        """
        fs           : input sampling rate (Hz)
        time_window_s: desired chunk length in seconds
        chunk_pad_s  : uniform extra seconds to pad before and after
        """
        self.fs = fs
        self.time_window_s = time_window_s
        self.chunk_pad_s = chunk_pad_s

        # Number of samples for ideal window and padding
        self.samples_per_window = int(np.floor(self.fs * self.time_window_s))
        self.pad_before = int(np.ceil(self.fs * self.chunk_pad_s))
        self.pad_after = int(np.ceil(self.fs * self.chunk_pad_s))

        # Create some fake data — 5 minutes long
        total_samples = int(fs * 300)
        self.time_axis = np.arange(total_samples) / fs
        self.data = np.sin(2*np.pi*1*self.time_axis)  # simple sine wave
        self.cursor = 0
        self.start_timestamp = pd.Timestamp("2024-01-01 00:00:00")

    def _get_next_chunk(self):
        required_samples = self.samples_per_window + self.pad_before + self.pad_after
        if self.cursor + required_samples > len(self.data):
            return None

        # Slice data with pad
        chunk_data = self.data[self.cursor : self.cursor + required_samples]
        chunk_time = self.time_axis[self.cursor : self.cursor + required_samples]
        chunk_timestamp = self.start_timestamp + pd.Timedelta(seconds=self.cursor / self.fs)

        # Advance cursor by *window length only*, keep right pad for overlap
        self.cursor += self.samples_per_window

        return {
            "trace": chunk_data,
            "time_axis": chunk_time,
            "timestamp": chunk_timestamp
        }

    def __iter__(self):
        return self

    def __next__(self):
        chunk = self._get_next_chunk()
        if chunk is None:
            raise StopIteration
        return chunk


# --- Run test ---
if __name__ == "__main__":
    fs_weird = 214.15982
    loader = TestLoader(fs=fs_weird, time_window_s=30, chunk_pad_s=0.05)

    chunks = list(loader)

    print(f"Total chunks: {len(chunks)}")

    for i, ch in enumerate(chunks[:3]):  # show first 3 chunks
        duration = ch["time_axis"][-1] - ch["time_axis"][0]
        print(f"\nChunk {i}:")
        print(f"Start timestamp: {ch['timestamp']}")
        print(f"Length in seconds (with pad): {duration:.6f}")
        print(f"First 5 samples: {ch['trace'][:5]}")
