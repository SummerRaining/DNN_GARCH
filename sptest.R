#load library
require(quantmod)
#first we download the daily time series data from Yahoo Finance
SP500 <- new.env()
getSymbols("^GSPC", env = SP500, src = "yahoo",
           from = as.Date("2012-01-04"),
           to = as.Date("2017-02-01"))
sp500 <- SP500$GSPC
head(sp500)
class(sp500)
dim(sp500)
chartSeries(sp500,theme="white")
sp500lr <-log(sp500$GSPC.Close+1)
#sp鎸囨暟鍋歀jung-box妫�楠?
Box.test(sp500lr,lag=10,type="Ljung")
#load rugarch library
require(rugarch)
spec = ugarchspec(variance.model = list(model = 'eGARCH', 
                    garchOrder = c(2, 1)), distribution = 'std')
setstart(spec) = list(shape = 5)
egarch1<- ugarchfit(spec, sp500lr[1:1000, , drop = FALSE], solver = 'hybrid')
egarch1
show(egarch1)
res = residuals(egarch1)
std_res = residuals(egarch1)/sigma(egarch1)  

#妫�楠屾ā鍨嬬殑arch鏁堝簲銆?
Box.test(res**2,lag = 10,type = 'Ljung')
Box.test(res,lag=10,type = 'Ljung')

#标准残差序列没有自相关性，也消除了arch效应。
Box.test(std_res**2,lag = 10,type = 'Ljung')
Box.test(std_res,lag=10,type = 'Ljung')
#write.csv(res,file="/Users/Apple/Desktop/姣曚笟鐨勯儴鍒嗘潗鏂?/fts-master/res.csv",row.names = F)
#write.csv(sigma(egarch1),file="/Users/Apple/Desktop/姣曚笟鐨勯儴鍒嗘潗鏂?/fts-master/sigma.csv",row.names = F)
