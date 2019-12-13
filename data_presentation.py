#先引入后面分析、可视化等可能用到的库
import tushare as ts
import pandas as pd  
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skew,kurtosis
import os


#正常显示画图时出现的中文和负号
from pylab import mpl
mpl.rcParams['font.sans-serif']=['SimHei']
mpl.rcParams['axes.unicode_minus']=False
#设置token
token='0779839291a3b8c2c578270f8cab10ccae52f4cf3868dfb3d60ed257'
#ts.set_token(token)
pro = ts.pro_api(token)
def data_analyse(x):
    print("mean :{:3f}\nMaximum :{:3f}\nMinimum :{:3f}\nstd.dev :{:3f}\n".format(np.mean(x),np.max(x),np.min(x),np.std(x)))
    
    print("\nskewness: {:3f}\nkurtosis: {:3f}".format(skew(x),kurtosis(x)))
    

if __name__ == "__main__":
# =============================================================================
#     if not os.path.exists("intermediate/shangzhengzonghe.csv"):
#         df = pro.index_daily(ts_code = "000001.SH")
#         df.to_csv("intermediate/shangzhengzonghe.csv",index = False)
#     else:
#         df = pd.read_csv("intermediate/shangzhengzonghe.csv")
# =============================================================================
    df = pro.index_daily(ts_code = "000001.SH")
    #将trade_date转换成时间格式的数据
    df.index = pd.to_datetime(df["trade_date"])
    df = df.sort_index()
    #截取1997-01-02到2019-12-01区间内的数据
    df = df[df["trade_date"]>"1997-01-02"]
    df = df[df["trade_date"]<"2019-12-01"]
    
    #画指数价格的图形。
    plt.figure(figsize=(8,3))
    plt.plot(df['close'])
    plt.title("上证综指")
    plt.savefig("plot_images/上证综指.png")
    plt.show()
    
    #转换成log收益率，画收益率图形。
    shdf = pd.DataFrame({'ratio':np.diff(np.log(df['close']))})
    shdf.index = df['trade_date'].iloc[1:].index
    shdf.head(5)
    print(shdf.describe())
    
    plt.figure(figsize=(8,3))
    plt.plot(shdf)
    plt.title("上证指数的收益率序列")
    plt.savefig("plot_images/上证收益率序列.png")
    plt.show()
    
    #得到统计量，这部分写入函数mean,max,min,std,skewness,
    data_analyse(shdf['ratio'].values)
    
    #绘制收益率的分布直方图    
    import seaborn
    plt.figure(figsize=(8,3))
    seaborn.distplot(shdf['ratio'].values,bins = 50,kde = False)
    plt.title("收益率分布直方图")
    plt.savefig("plot_images/收益率分布直方图.png")
    plt.show()
    plt.close()
    
    #平稳性检验，单位根检验。ADF
    from statsmodels.stats.diagnostic import unitroot_adf
    unitroot_adf(shdf['ratio'].values)
    
    #对序列做Ljung-box检验。
    import statsmodels as sm
    Q,P = sm.stats.diagnostic.acorr_ljungbox(shdf['ratio'].values,lags=20)
    box_test = pd.DataFrame({"Lags":np.arange(1,21),"Q_statistic":Q,"p_value":P})
    print(box_test)
    box_test.to_csv("intermediate/Ljung_box.csv")  # doctest: +SKIP

    
    
    
    
    