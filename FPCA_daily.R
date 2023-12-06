library(fda)
library(tidyr)
library(plotly)
library(stringr)


# Read and process data -----------------------------------------------------------------------

df <- read.csv2('data/load_with_calendar.csv', sep = ",", header = TRUE, row.names = "Date")

load_cols = c("Calabria", "Centre.North", "Centre.South", "North", "Sardinia", "Sicily",
              "South")

df[,load_cols] <- lapply(df[,load_cols], as.numeric)
head(df)

T <- nrow(df)
plot(1:T, df$Italy, type = "l")

library(tidyr)

# Pivot wider
df_fda <- df[,c(load_cols, "day", "daytype", "hour")]
df_fda <- pivot_wider(df_fda, id_cols = "hour", names_from = c(day, daytype), values_from = load_cols)
#df_fda <- df_fda[,colSums(!is.na(df_fda)) > 0]
df_fda <- df_fda[,colSums(is.na(df_fda)) == 0]
df_fda <- as.data.frame(df_fda)
rownames(df_fda) <- as.numeric(df_fda$hour)
df_fda$hour <- NULL
head(df_fda)

# Smoothing ----------------------------------------------------
T <- 23
t <- 0:T
df_fda_scaled <- scale(df_fda)

# Plot using matplot
#matplot(rownames(df_fda_scaled), df_fda_scaled[,], type = "l", col = 1:ncol(df_fda_scaled),
#        lty = 1, lwd = 2, xlab = "X", ylab = "Y",
#        main = "Daily load curves")

#basis <- create.fourier.basis(rangeval=range(1:T), nbasis=nbasis)

# Set parameters
m <- 5           # spline order 
degree <- m-1    # spline degree 
nbasis <- c(7, 9, 11, 13, 15, 17)

df.fd.list <- list()
for (p in nbasis) {
  #basis <- create.bspline.basis(rangeval=c(0,24), nbasis=p, norder=m)
  basis <- create.fourier.basis(rangeval=c(0,24), nbasis=p)
  df.fd <- Data2fd(y = as.matrix(df_fda_scaled), argvals = t, basisobj = basis)
  df.fd.list[[as.character(p)]] <- df.fd
}
#plot.fd(df.fd)

colors <- c("red", "green", "purple", "cyan")

show.random.smoothed.curve <- function(df_fda_scaled, nbasis, df.fd.list) {
  unit <- sample(colnames(df_fda), size = 1)
  
  plt <- plot_ly(
    x = t,
    y = df_fda_scaled[,unit],
    type = "scatter",
    mode = "markers+lines",
    name = "Original function",
  )
  
  k <- 1
  for (p in nbasis) {
    df.fd <- df.fd.list[[as.character(p)]]
    s.values <- eval.fd(t, df.fd)
    
    plt <- add_trace(
      plt,
      x = t,
      y = s.values[,unit],
      type = "scatter",
      mode = "lines",
      name = sprintf("Smoothed %d basis", p),
      line = list(color = colors[k])
    )
    k <- k+1
  }
  
  
  
  # Add a title to the plot
  plt <- plt %>% layout(
    title = unit,
    margin = list(t = 50),  # Set the top margin (adjust the value as needed),
    showlegend = TRUE  # You can customize other layout options as well
  )
  
  print(plt)
}

show.random.smoothed.curve(df_fda_scaled, nbasis, df.fd.list)

# Best number of basis is 11. How do these smoothed curves look?

nbasis <- 13
df.fd <- df.fd.list[[as.character(nbasis)]]
s.values <- eval.fd(seq(0, 24, 0.1), df.fd)

plot.full.smoothed.curve <- function(df_fda, s.values) {
  unit <- sample(colnames(df_fda), size = 1)
  
  plt <- plot_ly(
    x = seq(0, 24, 0.1),
    y = s.values[,unit],
    type = "scatter",
    mode = "lines",
    name = "Original function"
  )
  
  plt <- plt %>% layout(
    title = unit,
    margin = list(t = 50),  # Set the top margin (adjust the value as needed),
    showlegend = FALSE  # You can customize other layout options as well
  )
  print(plt)
}

plot.full.smoothed.curve(df_fda, s.values)

# FPCA ----------------------------------------------------------------------------------------
## Cumulative proportion of variance ----------------------------------------------------------
fpca <- pca.fd(df.fd, nharm=nbasis, centerfns=TRUE)
cumsum(fpca$varprop)

plot_ly(
  x = 1:nbasis,
  y = cumsum(fpca$varprop),
  type = "scatter",
  mode = "lines+markers",
) %>% layout(
  yaxis = list(range=c(0,1)),
  xaxis = list(range=c(1,13.1)),
  title = sprintf("Cumulative proportion of variance for %d basis", nbasis),
  margin = list(t = 50)
  )


#plot(cumsum(fpca$varprop), type='b', xlab='number of components', 
#     ylab='contribution to the total variance', ylim=c(0,1))


## Interpretation of the PCs ------------------------------------------------------------------
par(mfrow=c(1,2))
plot(fpca, nx=1000, pointplot=TRUE, harm=c(1,2,3,4), expand=0, cycle=FALSE)

plot(fpca$harmonics[1,],col=1,ylab='FPC1')
abline(h=0,lty=2)
plot(fpca$harmonics[2,],col=2,ylab='FPC2')

# It seems that the data is not so badly approximated in a 4-dimensional space (85% of the variance
# for the system with 13 basis functions)

## Scores -------------------------------------------------------------------------------------

df.scores <- fpca$scores
rownames(df.scores) <- colnames(df_fda)
df.scores <- as.data.frame(df.scores)
df.scores[,"month"] <- sapply(rownames(df.scores), function(x) str_extract(x, ".*-(.*)-.*", group=1), USE.NAMES = FALSE)
df.scores[,"region"] <- sapply(rownames(df.scores), function(x) str_extract(x, "(.*)_.*_.*", group=1), USE.NAMES = FALSE)
df.scores[,"daytype"] <- sapply(rownames(df.scores), function(x) str_extract(x, ".*_.*_(.*)", group=1), USE.NAMES = FALSE)

month <- "10"
daytype <- "Working day"
region <- "North"
df.plot <- df.scores[which((df.scores$month == month)&(df.scores$region == region)),]

plot_ly(
  x = df.plot[,1],
  y = df.plot[,2],
  type = "scatter",
  mode = "markers",
  #colors = "Blues",
  marker = list(size=8),
  color=df.plot[,"daytype"],
  text = rownames(df.plot)
) %>% layout(
  title = sprintf("Scores for month %s %s", month, daytype),
  margin = list(t = 50)
)

plot_ly(
  x = df.plot[,3],
  y = df.plot[,4],
  type = "scatter",
  mode = "markers",
  #colors = "Blues",
  marker = list(size=8),
  color=df.plot[,"region"],
  text = rownames(df.plot)
) %>% layout(
  title = sprintf("Scores for month %s %s", month, daytype),
  margin = list(t = 50)
)

pairs(df.plot[, 1:4], pch = 16, col = as.integer(as.factor(df.plot$daytype)), main = "Scatterplot Matrix")
legend("topright", legend = levels(df.scores$region), col = 1:7, pch = 16, title = "Regions")


## Approximation of the data in the projected space -------------------------------------------
pcbasis <- fpca$harmonics[1:4,]

s.values <- eval.fd(t, df.fd)

project.unit <- function(unit, nb.pc=4) {
  res <- fpca$meanfd
  for (k in 1:nb.pc) {
    res <- res + df.scores[unit,k] * fpca$harmonics[k,]
  }
  return(res)
}

unit <- sample(colnames(df_fda), size = 1)
plot(x=t, y=df_fda_scaled[,unit], lty=1, col='blue', lwd=2, main=unit, xlim = c(0,24))
lines(df.fd[unit,], lty=1, col='blue', lwd=2)
#lines(project.unit(unit, nb.pc = 1), col='red', main=unit, lwd=2)
#lines(project.unit(unit, nb.pc = 2), col='red', main=unit, lwd=2, lty=2)
lines(project.unit(unit, nb.pc = 4), col='red', main=unit, lwd=2, lty=3)
lines(project.unit(unit, nb.pc = 6), col='red', main=unit, lwd=2, lty=4)
lines(project.unit(unit, nb.pc = 8), col='red', main=unit, lwd=2, lty=5)
legend("topleft", legend=c("Original", "Smoothed", "1 PC", "2 PCs", "4 PCs", "6 PCs", "8 PCs"),
       col=c("blue", "blue", "red", "red", "red", "red", "red"),
       lty=c(1, 1, 1, 2, 3, 4, 5),
       lwd=2,
       title="Legend Title")

