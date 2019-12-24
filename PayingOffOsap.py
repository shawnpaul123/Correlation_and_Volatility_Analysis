'''

The following resources have been used in assistance of constructing of this program
-https://pythonprogramming.net/
-https://robotwealth.com/demystifying-the-hurst-exponent-part-2/
'''


'''
The program measures chaikin volaity , preprocesses 
data to feed into a LTSM to help predict future prices
based off an input of a basket of stocks(MFST,AAPL,IBM)
'''
#Create link for back-end to run on GPU
import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
#importing all libraries
import pandas as pd
import os
import quandl
import time
import pickle
import pandas as pd
from collections import deque
import random
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM,  BatchNormalization
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint, ModelCheckpoint
import time
from sklearn import preprocessing
from keras.utils import plot_model

auth_tok = #insert your token here-removed mine for safety#


#need to specify
comp_names  = ["WIKI/AAPL","WIKI/IBM","WIKI/MSFT"]
load_pickle = False
if load_pickle == True:
    for comp in comp_names:
            df = quandl.get(comp, trim_start = "2007-12-12", trim_end = "2016-12-30", authtoken=auth_tok)
            comp = comp.replace("/","_")
            pickle_out = open(comp+'.pickle','wb')
            pickle.dump(df,pickle_out)
            pickle_out.close()

#DATA COMPRESSION
pickle_in  = open("WIKI_AAPL"+".pickle","rb")
df_AAPL = pickle.load(pickle_in)

pickle_in  = open("WIKI_IBM"+".pickle","rb")
df_IBM = pickle.load(pickle_in)

pickle_in  = open("WIKI_MSFT"+".pickle","rb")
df_MSFT = pickle.load(pickle_in)

#Calculted key financial metrics for the 3 stocks (Expmovingaverage, chaikin vol hurst calculation(momentum predicted by oscillations around the accumulation-distribution line.), and)
#Financial daily metrics
def ExpMovingAverage(values,window,alpha,prev_value):
	sma = values/window
	val = sma*prev_value+(1-alpha)*sma
	return val


#%change in intrday prices
def percentChange(currentPoint, startPoint):
	#print(currentPoint)
	#print(startPoint)

	return ((float(currentPoint)-startPoint )/abs(startPoint))*100.0

#measures momentum predicted by oscillations around the accumulation-distribution line.
def chaikinVolValc(high,low,shift_high,shift_low):
	hml_new = high - low	
	hml_old = shift_high - shift_low

	highMlowEMA_new = ExpMovingAverage(hml_new,10,0.3,hml_old)
	highMlowEMA_old = ExpMovingAverage(hml_old,10,0.3,hml_old)

	cvc = percentChange(highMlowEMA_new, highMlowEMA_old)
	#print(cvc) 
	return cvc

#gets the hurst exponent
def hurst(p):  
    tau = []; lagvec = []  
    #  Step through the different lags  
    for lag in range(2,20):  
        #  produce price difference with lag  
        pp = np.subtract(p[lag:],p[:-lag])  
        
        #  Write the different lags into a vector  
        lagvec.append(lag)  
        #  Calculate the variance of the differnce vector  
        tau.append(np.sqrt(np.std(pp))) 

    #  linear fit to double-log graph (gives power)  
    m = np.polyfit(np.log10(lagvec),np.log10(tau),1)  
    # calculate hurst  
    hurst = m[0]*2  
    # plot lag vs variance  
    #py.plot(lagvec,tau,'o'); show()  
    return hurst 

##Create main df comprising of joined stock prices
def create_main_df():
    ratios = ["WIKI_MSFT","WIKI_IBM","WIKI_AAPL"]
      # the 4 ratios we want to consider
    for ratio in ratios:  # begin iteration
        #ratio = ratio.split('.csv')[0]  # split away the ticker from the file-name
        
        if ratio == "WIKI_MSFT":
            df = df_MSFT
        if ratio == "WIKI_IBM":
            df = df_IBM
        if ratio == "WIKI_AAPL":
            df = df_AAPL


        df['Date'] = df.index  
        
        columns_to_keep = ['Open', 'High', 'Low', 'Close', 'Volume', 'Date']
        df = df[columns_to_keep]
        #print(df.head())
        columns_rename_dict = {'Open': 'open', 'High':'high', 'Low':'low',
                               'Close':'close', 'Volume':'volume', 'Date':'time'}
        df = df.rename(columns=columns_rename_dict)

        main_df = pd.DataFrame()



        # rename volume and close to include the ticker so we can still which close/volume is which:
        df.rename(columns={"close": f"{ratio}_close", "volume": f"{ratio}_volume"}, inplace=True)
        
        df.set_index("time", inplace=True)  # set time as index so we can join them on this shared time
        df = df[[f"{ratio}_close", f"{ratio}_volume"]]  # ignore the other columns besides price and volume
        #print(df)
        if ratio == "WIKI_MSFT":
            MSFT = df
        if ratio == "WIKI_IBM":
            IBM = df
        if ratio == "WIKI_AAPL":
            AAPL = df
            

    df_outer = pd.concat([MSFT, AAPL], axis=1, sort=False)
    main_df  = pd.concat([df_outer, IBM], axis=1, sort=False)
    

     
    return main_df

def classify(current, future):
    if float(future) > float(current):  # if the future price is higher than the current, that's a buy, or a 1
        return 1
    else:  # otherwise... it's a 0!
        return 0

#scales values between 0-1
def preprocess_df(df):
    df = df.drop("future", 1)  # don't need this anymore.

    for col in df.columns:  # go through all of the columns
        if col != "target":  # normalize all ... except for the target itself!
            df[col] = df[col].pct_change()  
            df.dropna(inplace=True)  # remove the nas created by pct_change
            df[col] = preprocessing.scale(df[col].values)  # scale between 0 and 1.

    df.dropna(inplace=True)  


    sequential_data = []  # this is a list that will CONTAIN the sequences
    prev_days = deque(maxlen=SEQ_LEN)  # These will be our actual sequences. They are made with deque, which keeps the maximum length by popping out older values as new ones come in

    for i in df.values:  # iterate over the values
        prev_days.append([n for n in i[:-1]])  # store all but the target
        if len(prev_days) == SEQ_LEN:  # make sure we have 60 sequences!
            sequential_data.append([np.array(prev_days), i[-1]])  # append those bad boys!

    random.shuffle(sequential_data)  # shuffle for good measure.

    buys = []  # list that will store our buy sequences and targets
    sells = []  # list that will store our sell sequences and targets

    for seq, target in sequential_data:  # iterate over the sequential data
        if target == 0:  # if it's a "not buy"
            sells.append([seq, target])  # append to sells list
        elif target == 1:  # otherwise if the target is a 1...
            buys.append([seq, target])  # it's a buy!

    random.shuffle(buys)  # shuffle the buys
    random.shuffle(sells)  # shuffle the sells!

    lower = min(len(buys), len(sells))  # what's the shorter length?

    buys = buys[:lower]  # make sure both lists are only up to the shortest length.
    sells = sells[:lower]  # make sure both lists are only up to the shortest length.

    sequential_data = buys+sells  # add them together
    random.shuffle(sequential_data)  # another shuffle, so the model doesn't get confused with all 1 class then the other.

    X = []
    y = []

    for seq, target in sequential_data:  # going over our new sequential data
        X.append(seq)  # X is the sequenc 48es
        y.append(target)  # y is the targets/labels (buys vs sell/notbuy)

    return np.array(X), y  # return X and y...and make X a numpy array!



'''
Need to change df to take multiple dataframes - maybe throw in a commodity for prices
'''
#First part in hurst calculation in measuuring chaikin volaity and find its hurst exponent
hurst_val = False
if hurst_val == True:
    df = df_MSFT
    df['Date'] = df.index
    df = df.reset_index(level=0, drop=True).reset_index()

    df['High_Shift'] = df['High'].shift(10)
    df['Low_Shift'] = df['Low'].shift(10)
    

hurst_val_2 = False
if hurst_val_2 == True:
    df = df.drop(['index'],axis =1)
    df = df.dropna()
    df['CVC'] = df.apply(lambda x: chaikinVolValc(x['High'],x['Low'],x['High_Shift'],x['Low_Shift']),axis = 1)
    df = df.dropna()
    CVC_Hurst_List = df['CVC'].tolist()
    CVC_Hurst = hurst(CVC_Hurst_List)
    print(CVC_Hurst)
    
       




SEQ_LEN = 80 # preceeding sequence to collect for RNN
FUTURE_PERIOD_PREDICT = 2
RATIO_TO_PREDICT = "WIKI_MSFT"
EPOCHS = 1 # passes through our data
BATCH_SIZE = 64  # how many batches? Try smaller batch if you're getting OOM (out of memory) errors.
NAME = f"{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}"



# with 0s rather than NaNs


main_df_ = create_main_df()

main_df = main_df_
main_df.fillna(method="ffill", inplace=True)  # if there are gaps in data, use previously known values


main_df['future'] = main_df[f'{RATIO_TO_PREDICT}_close'].shift(-FUTURE_PERIOD_PREDICT)
main_df['target'] = list(map(classify, main_df[f'{RATIO_TO_PREDICT}_close'], main_df['future']))

main_df.dropna(inplace=True)

## split a specific part as future data.
times = sorted(main_df.index.values)
last_5pct = sorted(main_df.index.values)[-int(0.05*len(times))]

validation_main_df = main_df[(main_df.index >= last_5pct)]
main_df = main_df[(main_df.index < last_5pct)]

train_x, train_y = preprocess_df(main_df)
validation_x, validation_y = preprocess_df(validation_main_df)

print("hi")


print(f"train data: {len(train_x)} validation: {len(validation_x)}")
print(f"Dont buys: {train_y.count(0)}, buys: {train_y.count(1)}")
print(f"VALIDATION Dont buys: {validation_y.count(0)}, buys: {validation_y.count(1)}")

model = Sequential()
model.add(LSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.1))
model.add(BatchNormalization())

model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(2, activation='softmax'))



#
opt = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#Use NAG as it makes a jump based on its momentum first, and then  adjusts for the velocity, just like moi :)


# Compile model
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy']
)

# Train model
history = model.fit(
    train_x, train_y,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(validation_x, validation_y),
    #figure out checkpoints in keras
)

#plot_model(model, to_file='model.png')

# Score modelplot_model(model, to_file='model.png')
score = model.evaluate(validation_x, validation_y, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
