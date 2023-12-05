# ---- Imports ----
library(fda)
library(plotly)

# ---- Read data ----
df <- read.csv2('data/load.csv', sep = ",", header = TRUE, row.names = "Date")
plot(df$Italy, type='l')

# ---- Compute Periodogram ----
par(mfrow = c(2, 4))
for (colname in colnames(df)[c(-1)]) {
  # Compute the periodogram for the current column
  periodogram <- spec.pgram(as.numeric(df[grepl("2022", rownames(df)), colname]),
                            demean = TRUE, detrend = FALSE, plot = FALSE)
  # Plot the periodogram
  plot(periodogram$freq, periodogram$spec, type = "l", main = colname, xlab = "Frequency", ylab = "Squared Magnitude")
}

graphics.off()

# ---- Smooth using Fourier basis ----
year <- 2022
df <- read.csv2(sprintf('data/load_%d.csv', year), sep = ",", header = TRUE, row.names = "Date", )
df[] <- lapply(df, as.numeric)
head(df)

T <- nrow(df)
plot(1:T, df$Italy, type = "l")

## ---- GCV ----
# generalized cross-validation
nbasis <- seq(1, 3000, 100)
regions_gcv <- data.frame(row.names = nbasis)

for (region in colnames(df)){
  gcv <- numeric(length(nbasis))
  for (i in 1:length(nbasis)){
    basis <- create.fourier.basis(range(1:T), nbasis[i])
    gcv[i] <- smooth.basis(1:T, df[,region], basis)$gcv
  }
  regions_gcv[region] <- gcv
  #dev.off()
  #par(mfrow=c(1,1))
  plot(nbasis, gcv, main=region)
  cat(sprintf('The best number of basis for region %s is %d \n', region, nbasis[which.min(gcv)]))
  abline(v=nbasis[which.min(gcv)], col='red')
}

# Best number of basis seems to be about 2000



obs <- df$Italy
t <- 1:T
nbasis <- 2301

basis <- create.fourier.basis(rangeval=range(1:T), nbasis=nbasis)
#plot(basis)

s <- smooth.basis(argvals=t, y=obs, fdParobj=basis)
s.values <- eval.fd(t, s$fd) #  the curve smoothing the data

plt <- plot_ly(
  x = rownames(df),
  y = obs,
  type = "scatter",
  mode = "lines",
  name = "Original function",
)

add_trace(
  plt,
  x = rownames(df),
  y = s.values,
  type = "scatter",
  mode = "lines",
  name = "Smoothed function",
  line = list(color = "red")
)


plot(t, obs, xlab="t",ylab="observed data")
points(t, s.values, type="l", col="red", lwd=2)



# FPCA --------------------------------------------------------------------
matplot(df, type='l', main='Hourly Load (MW)', xlab='Hour', ylab='Consumption')

# Drop Italy
df <- df[, -which(names(df) == "Italy")]


## Creating data object ----------------------------------------------------
t <- 1:T
nbasis <- 3000

basis <- create.fourier.basis(rangeval=range(1:T), nbasis=nbasis)
df.fd <- Data2fd(y = as.matrix(df), argvals = t, basisobj = basis)
plot.fd(df.fd)

s.values <- eval.fd(t, df.fd) #  the curve smoothing the data

region <- "South"

plt <- plot_ly(
  x = rownames(df),
  y = df[,region],
  type = "scatter",
  mode = "lines",
  name = "Original function",
)

add_trace(
  plt,
  x = rownames(df),
  y = s.values[,region],
  type = "scatter",
  mode = "lines",
  name = "Smoothed function",
  line = list(color = "red")
)

plot(t, df[,region], xlab="t",ylab="observed data")
points(t, s.values[,region], type="l", col="red", lwd=2)

fpca <- pca.fd(df.fd, nharm=7, centerfns=TRUE)
fpca$varprop
par(mfrow=c(1,2))
plot(fpca, nx=1000, pointplot=TRUE, harm=c(1,2,3,4), expand=0, cycle=FALSE)

plot(fpca$harmonics[1,],col=1,ylab='FPC1',ylim=c(-0.1,0.08))
abline(h=0,lty=2)
plot(fpca$harmonics[2,],col=2,ylab='FPC2',ylim=c(-0.1,0.08))

# plot of the FPCs as perturbation of the mean
media <- mean.fd(df.fd)

plot(media, ylim=c(-5000,15000), ylab='temperature', main='FPC1')
lines(media + fpca$harmonics[1,] * sqrt(fpca$values[1]), col=2)
lines(media - fpca$harmonics[1,] * sqrt(fpca$values[1]), col=3)

plot(media, ylab='temperature', main='FPC2')
lines(media + fpca$harmonics[2,] * sqrt(fpca$values[2]), col=2)
lines(media - fpca$harmonics[2,] * sqrt(fpca$values[2]), col=3)

k <- 2
fmin <- media - fpca$harmonics[k,] * sqrt(fpca$values[k])
fplus <- media + fpca$harmonics[k,] * sqrt(fpca$values[k])

plt <- plot_ly(
  x = rownames(df),
  y = eval.fd(t, media),
  type = "scatter",
  mode = "lines",
  name = "Original function",
  line = list(color = "black")
)

line <- add_trace(
  plt,
  x = rownames(df),
  y = eval.fd(t, fmin),
  type = "scatter",
  mode = "lines",
  name = sprintf("Effect of negative score on PC%d",k),
  line = list(color = "red")
)

add_trace(
  line,
  x = rownames(df),
  y = eval.fd(t, fplus),
  type = "scatter",
  mode = "lines",
  name = sprintf("Effect of positive score on PC%d",k),
  line = list(color = "green")
)

plot(fpca$harmonics[k,])

