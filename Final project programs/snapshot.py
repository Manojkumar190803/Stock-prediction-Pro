import yfinance as yf

# Valid intervals: [1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo]')
def snapshot_data(folder, interval, filename='c:\\Users\manoj\Downloads\Major project data\Major pro source codes\DATASETS\indicesstocks.csv'):
    filepath='DATASETS/{}.csv'.format(filename)
    with open(filepath) as f:
        for line in f:
            if "," not in line:
                continue
            symbol = line.split(",")[0]
            data = yf.download(symbol, period='2y',interval=interval)
            ticketfilename = symbol.replace(".","_")
            data.to_csv('DATASETS/{}/{}.csv'.format(folder,ticketfilename))
            print("{} data for {} is downloaded sucessfully at {} ...!".format(interval, folder, symbol))
    return True

def snapshot_daily():
     #filepath='DATASETS/C:\\Users\manoj\Downloads\Major project data\DATASETS\Sectors & Stock symbols.csv'
     filepath="C:\\Users\manoj\Downloads\Major project data\Major pro source codes\DATASETS\indicesstocks.csv"
     with open(filepath) as f:
         for line in f:
             if "," not in line:
                 continue
             symbol = line.split(",")[1]
             symbols = line.split(",")
             for i in symbols:
                # #print(i)
                # #data = yf.download(symbol, start="1y", end="max")
                symbol=i
                try:
                    data = yf.download(symbol, period='5y',interval='1d')
                    ticketfilename = symbol.replace(".","_")
                    data.to_csv('C:\\Users\manoj\Downloads\Major project data\Major pro source codes\DATASETS\Daily_data/{}.csv'.format(ticketfilename))
                    print("script {} downloaded...!".format(symbol))
                except:
                    print("")
     return True

def snapshot_weekly():
     filepath="C:\\Users\manoj\Downloads\Major project data\Major pro source codes\DATASETS\indicesstocks.csv"
     with open(filepath) as f:
         for line in f:
             if "," not in line:
                 continue
             symbol = line.split(",")[1]
             symbols = line.split(",")
             for i in symbols:
                # #print(i)
                # #data = yf.download(symbol, start="1y", end="max")
                symbol=i
                try:
                    data = yf.download(symbol, period='5y',interval='1wk')
                    ticketfilename = symbol.replace(".","_")
                    data.to_csv('C:\\Users\manoj\Downloads\Major project data\Major pro source codes\DATASETS\Weekly_data/{}.csv'.format(ticketfilename))
                    print("script {} downloaded...!".format(symbol))
                except:
                    print("")
     return True

def snapshot_monthly():
     filepath="C:\\Users\manoj\Downloads\Major project data\Major pro source codes\DATASETS\indicesstocks.csv"
     with open(filepath) as f:
         for line in f:
             if "," not in line:
                 continue
             symbol = line.split(",")[1]
             symbols = line.split(",")
             for i in symbols:
                # #print(i)
                # #data = yf.download(symbol, start="1y", end="max")
                symbol=i
                try:
                    data = yf.download(symbol, period='5y',interval='1mo')
                    ticketfilename = symbol.replace(".","_")
                    data.to_csv('C:\\Users\manoj\Downloads\Major project data\Major pro source codes\DATASETS\Monthly_data/{}.csv'.format(ticketfilename))
                    print("script {} downloaded...!".format(symbol))
                except:
                    print("")
     return True

if __name__ == '__main__':
    snapshot_daily()
    snapshot_weekly()
    snapshot_monthly()
    #snapshot_data('DAILY_DATA','1d', 'INDICES')
    #snapshot_data('WEEKLY_DATA','1wk', 'INDICES')
    #snapshot_data('MONTHLY_DATA','1mo', 'INDICES')
    #snapshot_data('15min','15m', 'INDICES')
    #snapshot_data('DAILY','1d', 'INDICES')
    #snapshot_data('weekly','1wk', 'INDICES')
    #snapshot_data('monthly','1mo', 'INDICES')
    #snapshot_data('3months','3mo', 'INDICES')