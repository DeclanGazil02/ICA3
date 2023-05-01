# Import dependencies
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


# Step 1. Split the dataset into training and testing sets.
# Open main files
sub_folders = os.listdir('data_folder')

# open sub_folders containing wav
wavs = []  # list of lists
for folder in sub_folders:
    wavs.append(os.listdir('data_folder/' + str(folder)))


# split into specified emotions and then shuffle for randomness
angry = wavs[0]
fear = wavs[1]
happy = wavs[2]
sad = wavs[3]

# split into 70 train 30 test for each


# Step 2. Exploratory Data Analysis.
def time_freq_graphs(folder_expression, audio_file):
    # load input audio file
    signal, sample_rate = librosa.load('./data_folder/' + str(folder_expression) + '/' + str(audio_file))
    # plot audio files in time domain
    plt.figure(1)
    librosa.display.waveshow(y=signal, sr=sample_rate)
    plt.title('Time domain for ' + str(folder_expression) + ': ' + str(audio_file))
    plt.xlabel('Time / second')
    plt.ylabel('Amplitude')
    plt.show()

    # plot audio files in frequency domain
    k = np.arange(len(signal))
    T = len(signal) / sample_rate
    freq = k / T

    DATA_0 = np.fft.fft(signal)
    abs_DATA_0 = abs(DATA_0)
    plt.figure(2)
    plt.plot(freq, abs_DATA_0)
    plt.title('Freq domain for ' + str(folder_expression) + ': ' + str(audio_file))
    plt.xlabel("Frequency / Hz")
    plt.ylabel("Amplitude / dB")
    plt.xlim([0, 1000])
    plt.show()

    # plot the time-frequency variation of the audio
    D = librosa.stft(signal)  # STFT of y
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    plt.figure(3)
    plt.title('Time-Freq Variation for ' + str(folder_expression) + ': ' + str(audio_file))
    librosa.display.specshow(S_db, x_axis='time', y_axis='log')
    plt.colorbar()
    plt.show()


# plot for random audio file in the training datasets
time_freq_graphs('angry', angry[12])
time_freq_graphs('fear', fear[25])
time_freq_graphs('happy', happy[50])
time_freq_graphs('sad', sad[39])


# Step 3. Acoustic Feature Extraction
def feature_extraction(folder_expression, wav_list):
    master_df = []

    for audio_file in wav_list:
        # load first
        signal, sample_rate = librosa.load('./data_folder/' + str(folder_expression) + '/' + str(audio_file))

        # extract loudness
        df_loudness = pd.DataFrame()
        S, phase = librosa.magphase(librosa.stft(signal))
        rms = librosa.feature.rms(S=S)
        df_loudness['Loudness'] = rms[0]

        # extract mel-frequency cepstral coefficients
        df_mfccs = pd.DataFrame()
        mfccs = np.mean(librosa.feature.mfcc(y=signal, sr=sample_rate).T, axis=0)
        df_mfccs['MFCC'] = mfccs

        # extract zero crossing rate
        df_zero_crossing_rate = pd.DataFrame()
        zcr = librosa.feature.zero_crossing_rate(y=signal)
        df_zero_crossing_rate['ZCR'] = zcr[0]

        # extract chroma
        df_chroma = pd.DataFrame()
        chroma = np.mean(librosa.feature.chroma_stft(y=signal, sr=sample_rate).T, axis=0)
        df_chroma['Chroma'] = chroma

        # extract mel spectrogram
        df_mel_spectrogram = pd.DataFrame()
        mel = np.mean(librosa.feature.melspectrogram(y=signal, sr=sample_rate).T, axis=0)
        df_mel_spectrogram['Mel_Spectrogram'] = mel

        # combine features into one dataframe
        feature_matrix = pd.concat([df_loudness, df_mfccs, df_zero_crossing_rate, df_chroma, df_mel_spectrogram],
                                   axis=1)
        master_df.append(feature_matrix)
    return master_df


angry_df = feature_extraction('angry', angry)
fear_df = feature_extraction('fear', fear)
happy_df = feature_extraction('happy', happy)
sad_df = feature_extraction('sad', sad)

# Step 4. Feature Post-processing

# average features
def average_features(list_of_df):
    new_df = []
    for df in list_of_df:
        # add averaged value to a temp array then assign to a dataframe for each
        averaged_df = pd.DataFrame()
        temp = []
        temp.append(df['Loudness'].mean())
        averaged_df['Loudness'] = temp

        temp = []
        temp.append(df['MFCC'].mean())
        averaged_df['MFCC'] = temp

        temp = []
        temp.append(df['ZCR'].mean())
        averaged_df['ZCR'] = temp

        temp = []
        temp.append(df['Chroma'].mean())
        averaged_df['Chroma'] = temp

        temp = []
        temp.append(df['Mel_Spectrogram'].mean())
        averaged_df['Mel_Spectrogram'] = temp

        new_df.append(averaged_df)
    return new_df


# set average
angry_df = average_features(angry_df)
fear_df = average_features(fear_df)
happy_df = average_features(happy_df)
sad_df = average_features(sad_df)


def set_emotion(folder_expression, df_list):
    df_w_emotion = []
    for df in df_list:
        emotion_df = pd.DataFrame()
        if folder_expression == 'angry':
            emotion_num = 0
        elif folder_expression == 'fear':
            emotion_num = 1
        elif folder_expression == 'happy':
            emotion_num = 2
        else:
            emotion_num = 3
        emotion_df['Emotion'] = np.full(len(df['Loudness']), int(emotion_num))
        df = pd.concat([df, emotion_df], axis=1)
        df_w_emotion.append(df)
    return df_w_emotion


# set emotion type
angry_df = set_emotion('angry', angry_df)
fear_df = set_emotion('fear', fear_df)
happy_df = set_emotion('happy', happy_df)
sad_df = set_emotion('sad', sad_df)


x = []
y = []

# add all necessary values for x and y
for df in angry_df:
    x.append(df.loc[0].values)
    y.append(df['Emotion'])

for df in fear_df:
    x.append(df.loc[0].values)
    y.append(df['Emotion'])


for df in happy_df:
    x.append(df.loc[0].values)
    y.append(df['Emotion'])


for df in sad_df:
    x.append(df.loc[0].values)
    y.append(df['Emotion'])

x = np.array(x)

# Reshape x to be a 2D array of shape (400, 6)
x = np.reshape(x, (400, 6))

y = np.array(y)
y = y.ravel()

# use .3 test size for 70 30 split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3, random_state=9)



def print_metrics(accuracy, report):
    print('Accuracy:', accuracy)
    print(report)
    print()


# SVC
svc = SVC(kernel='linear', C=1, probability=True)
svc.fit(x_train, y_train)

y_pred = svc.predict(x_test)
emotion_accuracy = accuracy_score(y_test, y_pred)
emotion_report = classification_report(y_test, y_pred)

# Print the metrics
print('\n', 'SVC: ', '\n')
print_metrics(emotion_accuracy, emotion_report)

# NB
nbc = GaussianNB()
nbc.fit(x_train, y_train)

y_pred = nbc.predict(x_test)
emotion_accuracy = accuracy_score(y_test, y_pred)
emotion_report = classification_report(y_test, y_pred)

# Print the metrics
print('\n', 'NBC: ', '\n')
print_metrics(emotion_accuracy, emotion_report)

# Random Forest
rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)

y_pred = rfc.predict(x_test)
emotion_accuracy = accuracy_score(y_test, y_pred)
emotion_report = classification_report(y_test, y_pred)

# Print the metrics
print('\n', 'RFC: ', '\n')
print_metrics(emotion_accuracy, emotion_report)


