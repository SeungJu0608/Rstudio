

\#\#1. Cheongpa2 자료를 활용하여 최근 7일간 체류인구로부터 내일의 체류인구를 예측한다.

\*\* 목표  
\- 예측을 위한 지도학습 모형 구축의 전 과정을 구현하기 ( 훈련, 비교, 선택, 테스트 )

### 문제 해결을 위한 라이브러리 불러오기

``` r
library(class)
library(FNN)
```

    ## 
    ## Attaching package: 'FNN'

    ## The following objects are masked from 'package:class':
    ## 
    ##     knn, knn.cv

``` r
library(glmnet)
```

    ## Loading required package: Matrix

    ## Loaded glmnet 4.0-2

``` r
library(ggplot2)
```

### 데이터 불러오기

``` r
load("Cheongpa2.Rdata")
head(Cheongpa2)
```

    ##         Date Today  Lag1  Lag2  Lag3  Lag4  Lag5  Lag6  Lag7
    ## 1 2018-12-08 19785 26437 27519 27105 26583 26618 22751 24606
    ## 2 2018-12-09 20630 19785 26437 27519 27105 26583 26618 22751
    ## 3 2018-12-10 26257 20630 19785 26437 27519 27105 26583 26618
    ## 4 2018-12-11 25284 26257 20630 19785 26437 27519 27105 26583
    ## 5 2018-12-12 25862 25284 26257 20630 19785 26437 27519 27105
    ## 6 2018-12-13 25054 25862 25284 26257 20630 19785 26437 27519

### 데이터 설명

  - 예측대상(response, target) : \(y_{i}\) (변수이름 Today)  
  - 피처(feature, predictor) :
    \(x_{i1}(Lag1), x_{i2}(Lag2) ... x_{i7}(Lag7)\)
  - target data와 feature 변수는 수치형 데이터이며 날짜 형식의 데이터인 Date 변수로 데이터셋이 이루어져
    있다.
  - target data를 연속형 확률변수(numeric)로 간주하여 회귀문제로 생각한다.  
  - 데이터의 NA값과 NULL값이 존재하지 않아 Missing Value값이 없다는 것을 확인할 수 있다.
  - 아래의 시계열 그래프를 확인해 보았을 때 이상치로 의심되는 값들이 존재하는지 확인해 볼 수 있다. 우선

<!-- end list -->

``` r
sum(is.null(Cheongpa2))
```

    ## [1] 0

``` r
sum(is.na(Cheongpa2))
```

    ## [1] 0

``` r
ggplot(data = Cheongpa2) + 
  geom_line(aes(x = Date, y = Today), color="red") + 
  geom_line(aes(x = Date, y = Lag1), color="yellow") + 
  geom_line(aes(x = Date, y = Lag2), color="gold") + 
  geom_line(aes(x = Date, y = Lag3), color="green") + 
  geom_line(aes(x = Date, y = Lag4), color="darkgreen") +
  geom_line(aes(x = Date, y = Lag5), color="skyblue") +
  geom_line(aes(x = Date, y = Lag6), color="blue") + 
  geom_line(aes(x = Date, y = Lag7), color="purple") 
```

![](timeseries_files/figure-gfm/unnamed-chunk-4-1.png)<!-- --> 2018년 12월
데이터와 2019년 데이터를 한 그룹으로 2020년 데이터를 하나의 그룹으로 묶어 시계열 데이터를 보았을 때 다음과 같다

``` r
data_year = Cheongpa2
data_year$year = ifelse(as.character(data_year$Date, "%Y") == 2020, "2", "1")
breaks = seq(as.Date("2018-12-01"),as.Date("2020-12-31"), by="1 month" )
ggplot(data = data_year, aes(x=Date)) + geom_point(data = data_year, aes(y=Today, color=year)) + scale_x_date(breaks = breaks) +theme(axis.text.x = element_text(angle=30, hjust=1))
```

![](timeseries_files/figure-gfm/unnamed-chunk-5-1.png)<!-- --> 2019년 8월
말의 데이터 3개가 현저히 낮은 값이며 2019-10월의 2개의 데이터가 상대적으로 큰 값을 지녀 이상값이라고 의심을 할 수
있다. 시계열 데이터의 outlier을 찾아주는 tsoutliers 패키지의 tsoutliers함수를 이용하여 이상치를 확인해
보도록 한다.

``` r
library(fpp)
```

    ## Loading required package: forecast

    ## Registered S3 method overwritten by 'quantmod':
    ##   method            from
    ##   as.zoo.data.frame zoo

    ## Loading required package: fma

    ## Loading required package: expsmooth

    ## Loading required package: lmtest

    ## Loading required package: zoo

    ## 
    ## Attaching package: 'zoo'

    ## The following objects are masked from 'package:base':
    ## 
    ##     as.Date, as.Date.numeric

    ## Loading required package: tseries

``` r
timeset=ts(Cheongpa2[,2], start = c(2018,12), frequency = 365.25) # 연간 계절성으로 이용
tsoutliers(timeset)
```

    ## $index
    ## [1] 254 260 261
    ## 
    ## $replacements
    ## [1] 18093.50 18358.67 19212.33

``` r
print(Cheongpa2[c(254,260,261),c(1,2)])
```

    ##           Date Today
    ## 254 2019-08-18 14136
    ## 260 2019-08-24 11745
    ## 261 2019-08-25 13908

연간 주기성으로 데이터를 이용하여 tsoutliers함수의 결과값 replacement값을 이용한다. 이함수에서
replacement 값은 ARIMA를 이용하는 것으로 생각된다. 2019년 8월 18,24,25일의 3개의 관측값이 이상치로
각각 18302.14, 19255.86, 19608.83 값으로 대체하여 다음 문제를 해결해 나가기로 한다.

이때 Lag1에서 Lag7값은 시차에 관한 데이터 값이므로 이를 주의하여 데이터값을 대체한다.

``` r
data_re = Cheongpa2
for ( i in 1:8){
   data_re[c(254+(i-1),260+(i-1),261+(i-1)),i+1] = c(18093.50, 18358.67, 19212.33)
}
```

### 데이터셋 나누기 (tr : val : te = 6:2:2)

  - 위 데이터는 시간에 따른 시계열 데이터이므로 날짜의 흐름에 따른 훈련데이터,검증데이터, 학습데이터의 분할이 요구된다.

<!-- end list -->

``` r
x = model.matrix(Today~., data=data_re)[,-c(1,2)]
y = data_re$Today

tr = 0.6
va = 0.2
n = nrow(data_re)

x.train = x[1 : floor(n*tr),]
y.train = y[1 : floor(n*tr)]
x.val = x[(floor(n*tr) + 1 ) : floor(n*(tr + va)), ]
y.val = y[(floor(n*tr) + 1 ) : floor(n*(tr + va)) ]
x.test = x[(floor(n*(tr + va)) + 1 ): n,]
y.test = y[(floor(n*(tr + va)) + 1 ) : n]
```

### 5개 모형과 3가지 오류측도 with validation data set (RMSE, MAE, MAPE)

  - 3가지 오류측도는 시계열 데이터임을 감안하여 RMSE, MAE, MAPE로 지정하였다.
  - \(RMSE = \sqrt{MSE}\) \(MAE = ave(|y-\widehat{y}|)\)
    \(MAPE = ave(|\frac{y-\widehat{y}}{y}|)*100\)
  - 각 모형을 훈련데이터로 적합시키고 validation error가 가장 낮은 모형을 선택하여 각각의 오류 지표들을
    비교한다.

#### (1) naive benchmark : y의 평균

``` r
yhat.naive = mean(y.train) 

rmse.naive = sqrt( mean( (y.val-yhat.naive)^2 ) )
mae.naive = mean( abs( y.val-yhat.naive ) )
mape.naive = mean( abs( (y.val-yhat.naive)/y.val) )*100

val.err.naive = c(rmse.naive, mae.naive, mape.naive )
```

#### (2) k-최근접이웃 regression

  - K는 20으로 지정하였으며 1부터 20까지의 K를 적용하여 k-최근접이웃 모형을 적합하였다.

<!-- end list -->

``` r
K=20
rmses.knn = rep(NA,K)
maes.knn = rep(NA,K)
mapes.knn = rep(NA,K)
for (i in 1:K) {
  obj.knn = knn.reg(train = x.train, test = x.val, y = y.train, k=i) 
  yhat.knn = obj.knn$pred
  rmses.knn[i] = sqrt(mean((y.val- yhat.knn)^2))
  maes.knn[i] = mean(abs(y.val-yhat.knn))
  mapes.knn[i] = mean(abs((y.val-yhat.knn)/y.val))*100
}
plot(x=1:K, y=rmses.knn, col='red')
```

![](timeseries_files/figure-gfm/unnamed-chunk-11-1.png)<!-- -->

``` r
plot(x=1:K, y=maes.knn, col='green')
```

![](timeseries_files/figure-gfm/unnamed-chunk-11-2.png)<!-- -->

``` r
plot(x=1:K, y=mapes.knn, col='blue')
```

![](timeseries_files/figure-gfm/unnamed-chunk-11-3.png)<!-- -->

``` r
k.optimal.rmse = (1:K)[which.min(rmses.knn)]
k.optimal.mae = (1:K)[which.min(maes.knn)]
k.optimal.mape = (1:K)[which.min(mapes.knn)]
rmse.knn = rmses.knn[which.min(rmses.knn)]
mae.knn = maes.knn[which.min(maes.knn)]
mape.knn = mapes.knn[which.min(mapes.knn)]

val.err.knn =c( "RMSE.knn.7"=rmse.knn, "MAE.knn.7"=mae.knn, "MAPE.knn.7"=mape.knn  )
val.err.knn
```

    ##  RMSE.knn.7   MAE.knn.7  MAPE.knn.7 
    ## 1456.986561 1058.577922    3.447807

  - 위의 결과로 1부터 20까지의 K 중 세가지 오류측도모두가 가장 낮은 k=7을 선택하여 다른 모형들의 오류측도와 비교한다.
    이때의 오류측도들은 위와 같다.

#### (3) 선형회귀

  - 반응변수를 Today로 하여 Lag1\~Lag7과 다중선형회귀모형으로 적합시킨다.

<!-- end list -->

``` r
obj.lm = lm(y.train ~ x.train) 
yhat.lm = cbind(1,x.val)%*%coef(obj.lm)
rmse.lm = sqrt(mean((y.val-yhat.lm)^2))
mae.lm = mean(abs(y.val-yhat.lm))
mape.lm = mean(abs((y.val-yhat.lm)/y.val))*100

val.err.lm = c(rmse.lm, mae.lm, mape.lm )
```

#### (4) Ridge Regression

  - 능형회귀의 하이퍼파라미터값인 람다값을 \(2^{-30}\) 부터 \(2^{30}\) 까지의 값을 100개의 등구간으로
    나누어 고려한다.

<!-- end list -->

``` r
grid = 2^seq(from=30, to=-30, length=100) 

obj.ridge = glmnet(x=x.train, y=y.train, family = "gaussian", alpha = 0, lambda = grid,standardize=T)
yhat.ridge = predict(obj.ridge, newx = x.val, type = "link")

rmses.ridge = apply(yhat.ridge, 2, function(x) sqrt(mean((y.val-x)^2)))
maes.ridge = apply(yhat.ridge, 2, function(x) mean(abs(y.val-x)))
mapes.ridge = apply(yhat.ridge, 2, function(x) mean(abs((y.val-x)/y.val))*100)

rmse.ridge = rmses.ridge[which.min(rmses.ridge)]
mae.ridge = maes.ridge[which.min(maes.ridge)]
mape.ridge = mapes.ridge[which.min(mapes.ridge)]
c("lambda.rmse"=grid[which.min(rmses.ridge)], "lambda.mae"=grid[which.min(maes.ridge)], "lambda.mape"=grid[which.min(mapes.ridge)])
```

    ##  lambda.rmse   lambda.mae  lambda.mape 
    ## 1.534178e+01 9.313226e-10 9.313226e-10

  - 람다에 대한 오류측도 시각화 그래프

<!-- end list -->

``` r
plot(x=log2(obj.ridge$lambda), y=rmses.ridge,ylab='validation error',type="o",col="red")
```

![](timeseries_files/figure-gfm/unnamed-chunk-14-1.png)<!-- -->

``` r
plot(x=log2(obj.ridge$lambda), y=maes.ridge,pch=4,type="o",col="blue")
```

![](timeseries_files/figure-gfm/unnamed-chunk-14-2.png)<!-- -->

``` r
plot(x=log2(obj.ridge$lambda), y=mapes.ridge,pch=2,col="darkgreen")
```

![](timeseries_files/figure-gfm/unnamed-chunk-14-3.png)<!-- -->

``` r
val.err.ridge = c(rmses.ridge[100], mae.ridge, mape.ridge)
lambda.optimal.mse.ridge = grid[which.min(maes.ridge)]
yhat.ridge.opt = yhat.ridge[,which.min(maes.ridge)]
```

  - 이때 각각의 오류측도를 가장 낮게 하는 람다의 값은 세가지 모두 달랐다. 그 중100번째 람다값(9.313226e-10)이
    MAE와 MAPE 두개의 검증오류측도를 가장 낮게 만들었기 때문에 이 그리드 값을 기준으로 한 세 오류측도를 이용해
    비교하고 test데이터에 적합할 것이다.

#### (5) Lasso Regression

``` r
obj.lasso = glmnet( x=x.train, y=y.train, family="gaussian", alpha = 1, lambda = grid)
yhat.lasso = predict(obj.lasso, newx = x.val, type = "link")

rmses.lasso = apply(yhat.lasso, 2, function(x) sqrt(mean((y.val-x)^2)))
maes.lasso = apply(yhat.lasso, 2, function(x) mean(abs(y.val-x)))
mapes.lasso = apply(yhat.lasso, 2, function(x) mean(abs((y.val-x)/y.val))*100)

rmse.lasso = rmses.lasso[which.min(rmses.lasso)]
mae.lasso = maes.lasso[which.min(maes.lasso)]
mape.lasso = mapes.lasso[which.min(mapes.lasso)]
c("lambda.rmse"=grid[which.min(rmses.lasso)], "lambda.mae"=grid[which.min(maes.lasso)], "lambda.mape"=grid[which.min(mapes.lasso)])
```

    ##  lambda.rmse   lambda.mae  lambda.mape 
    ## 3.498597e-01 9.313226e-10 2.192056e-07

  - 람다에 대한 오류측도 시각화 그래프

<!-- end list -->

``` r
plot(x=log2(obj.lasso$lambda), y=rmses.lasso,ylab='validation error',type="o",col="red")
```

![](timeseries_files/figure-gfm/unnamed-chunk-17-1.png)<!-- -->

``` r
plot(x=log2(obj.lasso$lambda), y=maes.lasso,pch=4,type="o",col="blue")
```

![](timeseries_files/figure-gfm/unnamed-chunk-17-2.png)<!-- -->

``` r
plot(x=log2(obj.lasso$lambda), y=mapes.lasso,pch=2,col="darkgreen")
```

![](timeseries_files/figure-gfm/unnamed-chunk-17-3.png)<!-- -->

``` r
val.err.lasso = c(rmses.lasso[100], mae.lasso, mapes.lasso[100])
lambda.optimal.lasso = grid[which.min(maes.lasso)]
yhat.lasso.opt = yhat.lasso[,which.min(maes.lasso)]
```

  - (4)의 능형회귀와 마찬가지로 세가지 오류측도를 모두 가장 낮게 하는 람다를 기준으로 한다. 그러나 위의 결과 세가지
    오류측도를 가장 낮게 하는 람다는 각각 달랐다. 따라서 능형회귀에서와 동일한 과정을 따라 MAE를 기준으로 한
    것을 기준으로 적용하여 95번째 람다값을 최적 람다값이라고 지정하였다.

### 모형들의 검증세트오류 시각화

  - 표 - <검증오류 비교>

<!-- end list -->

``` r
val.err.mat = data.frame(val.err.naive, val.err.knn, val.err.lm, val.err.ridge, val.err.lasso, row.names=c("RMSE","MAE", "MAPE"))
val.err.mat
```

    ##      val.err.naive val.err.knn  val.err.lm val.err.ridge val.err.lasso
    ## RMSE    4999.97957 1456.986561 1205.216513   1205.216513   1205.216514
    ## MAE     4752.22220 1058.577922  922.746034    922.746034    922.746034
    ## MAPE      14.96257    3.447807    2.973925      2.973925      2.973925

  - \(y_{validation}\) 과 \(\widehat{y}_{validation}\) 의 시각화

<!-- end list -->

``` r
plot(x=Cheongpa2$Date[(floor(n*tr) + 1 ) : floor(n*(tr + va))], y=y.val, xlab = "Date in Validation set")
abline(h=yhat.naive)
points(x=Cheongpa2$Date[(floor(n*tr) + 1 ) : floor(n*(tr + va))], y=yhat.knn, pch="*", col="red")
points(x=Cheongpa2$Date[(floor(n*tr) + 1 ) : floor(n*(tr + va))], y=yhat.lm, pch="*", col="blue")
points(x=Cheongpa2$Date[(floor(n*tr) + 1 ) : floor(n*(tr + va))], y=yhat.ridge.opt, pch="o", col="darkgreen")
points(x=Cheongpa2$Date[(floor(n*tr) + 1 ) : floor(n*(tr + va))], y=yhat.lasso.opt, pch="^", col="purple")
legend("bottomright", c( "knn", "lm", "ridge", "lasso") , col=c("red","blue","darkgreen","purple"), pch = c("*","*","o","^"))
```

![](timeseries_files/figure-gfm/unnamed-chunk-20-1.png)<!-- -->

### 최종모형의 선택 및 적합

  - 위의 y와 \(\widehat{y}\)에 대한 그래프를 통해 직관적으로 보았을때 능형회귀와 라쏘회귀로 적합된
    \(\widehat{y}\)값들이 얼추 비슷하며 이 두 방법을 통한 적합의 성능이 더 좋아보인다.
  - 위의 표 <검증오류 비교 >에서 각 검증오류측도를 모두 최소화하는 모형은 능형회귀모형이며 각 감소정도는 mse기준이 가장
    두드러진다.
  - 따라서 최종모형은 능형회귀로한다.

<!-- end list -->

``` r
x.final = rbind(x.train,x.val)
y.final = c(y.train, y.val)

obj.final.ridge = glmnet( x=x.final, y=y.final, family="gaussian", standardize = T, lambda = lambda.optimal.mse.ridge)
yhat.final.ridge = predict(obj.final.ridge, newx = x.test, type="link")

plot(x=Cheongpa2$Date[(length(y.final)+1):n], y=y.test, xlab = "Date in final data")
points(x=Cheongpa2$Date[(length(y.final)+1):n], y=yhat.final.ridge ,pch="*", col="red")
legend("bottomright", c( "y", "yhat") , col=c("black","red"), pch = c("o","*"))
```

![](timeseries_files/figure-gfm/unnamed-chunk-21-1.png)<!-- --> \*\* y 와
\(\widehat{y}\) 의 시각화 그래프를 통해서도 잘 적합이 된 상태를 볼 수 있다.\*\*
