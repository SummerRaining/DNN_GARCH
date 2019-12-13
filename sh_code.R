#read data,and slit data of 1997-01-1
data = read.csv('/Users/a/Desktop/DNN-ARCH project/intermediate/shanghai.csv')
#as.character将数据类型转换为字符串类型。str(x)可以用于查看变量的类型。
data$trade_date = as.Date(as.character(data$trade_date),"%Y%m%d")
#time类型的变量可以和对应的字符串类型直接比较。
data = data[data$trade_date> "1997-01-02",]
data = data[data$trade_date< "2019-12-01",]

#将收盘价转换成收益率序列。
data['log_close'] = log(data['close'])
ratio = diff(data$log_close)
df.ratio = data.frame("ratio" = ratio,'trade_date' = data$trade_date[1:dim(data)[1]-1])

require(fGarch)
require(FinTS)
library("tseries")
#检验正态分布
jarque.bera.test(df.ratio$ratio)
sd(df.ratio$ratio)

#自相关系数
acf(df.ratio$ratio,lag = 20)
pacf(df.ratio$ratio,lag = 20)

#ljung-box检验相关性
for(i in 1:20){
  result = Box.test(df.ratio$ratio,lag=i,type="Ljung");
  cat("\n",i,"阶的统计量：",as.character(result[1]),"  p值为",as.character(result[3]));
  
}

#检验序列的平稳性,adf检验序列的平稳性。
adf.test(df.ratio$ratio)

#arch效应检验
#去除序列中相关性,此时可以认为序列不相关了。
m1 = arima(df.ratio$ratio, order = c(3, 0, 2))
m1
Box.test(m1$residuals, lag = 10, type = 'Ljung')
#检验残差的平方序列是否自相关，或者直接检验序列中的arch效应。
Box.test(m1$residuals^2, lag = 10, type = 'Ljung')
tarch = ArchTest(df.ratio$ratio, lag=10)
tarch


#确定最优参数的函数
#根据aic赤池准则，
best_param = function(x){
  best_aic = Inf
  best_alpha = 1
  best_beta = 1
  for(a in 1:7){
    for(b in 1:7){
      model = garchFit(substitute(~ garch(p,q),list(p=a, q=b)), data = x, trace = F)
      aic = model@fit[["ics"]][["AIC"]]
      cat("alpha is\t",a,"beta is\t",b,"aic is \t",aic,'\n\n')
      if(aic<best_aic){
          best_alpha = a;
          best_beta = b;
          best_aic = aic;
      }
    }
  }
  return(c(best_alpha,best_beta));
}
params = best_param(m1$residuals)
params = c(1,1)
cat("garch model,best alpha is:",params[1],"\tbest beta is:",params[2],'\n')

#建立garch模型,m2中的参数可以用@取出来
shgarch = garchFit(~ garch(1, 1), data = m1$residuals, trace = F)
summary(shgarch)

#检验标准残差中的arch效应
std_res = residuals(shgarch,standardize = TRUE)
atest = ArchTest(std_res)
cat("garch's best aic is:",shgarch@fit[["ics"]][["AIC"]],'\n')
cat("garch's std_res arch test p value:",atest$p.value[1],'\n')

#寻找egarch的最优参数
egarch_best_param = function(x){
  best_aic = Inf
  best_alpha = 1
  best_beta = 1
  for(a in 1:7){
    for(b in 1:7){
      spec = ugarchspec(variance.model = list(model = 'eGARCH', 
                       garchOrder = c(a, b)), distribution = 'std')
      setstart(spec) = list(shape = 5)
      egarch1<- ugarchfit(spec, m1$residuals, solver = 'hybrid')
      aic = infocriteria(egarch1)[1]
      cat("alpha is\t",a,"beta is\t",b,"aic is \t",aic,'\n')
      if(aic<best_aic){
        best_alpha = a;
        best_beta = b;
        best_aic = aic;
      }
    }
  }
  cat("best params alpha is:",best_alpha,"\tbeta is:",best_beta,"best aic is:",best_aic,'\n')
  return(c(best_alpha,best_beta));
}

#使用egarch模型分析,打印模型的aic值。与Archtest检验的p值。
params = egarch_best_param(m1$residuals)
params = c(1,1)
spec = ugarchspec(variance.model = list(model = 'eGARCH',garchOrder = params),
                  mean.model=list(armaOrder=c(0,0), include.mean=TRUE),
                  distribution = 'std')
setstart(spec) = list(shape = 5)
egarch1<- ugarchfit(spec, m1$residuals, solver = 'hybrid')

std_res = residuals(egarch1)/sigma(egarch1)
atest = ArchTest(std_res)
cat("egarch's aic is:",infocriteria(egarch1)[1],"\n")
cat("egarch' std_res arch test p value:",atest$p.value[1],'\n')

#将残差导出
write.csv(res,file="/Users/a/Desktop/DNN-ARCH project/intermediate/arma_residual.csv",row.names = F)
