library(fda)
library(tidyr)
library(plotly)
library(stringr)
library(progress)
library(lubridate)

# Read and process data -----------------------------------------------------------------------

input_df <- read.csv2('data/1_input/load/load_with_calendar.csv', sep = ",", header = TRUE, row.names = "Date")

# HOTFIX: Add row for 00:00 of the day following the last day in the df
input_df_2023 <- read.csv2("data/1_input/load/refresh_202402/load_2023_with_calendar.csv", sep = ",",
                           header = TRUE, row.names = "Date")
input_df <- rbind(input_df, input_df_2023[1,])

# HOTFIX: Add value for Calabria 2021-01-01 00:00
input_df['2021-01-01 00:00:00', 'Calabria'] <- 620

# HOTFIX: handle na values for hour loss days 
hour_loss_days = c(
  "2018-03-25",
  "2019-03-31",
  "2020-03-29",
  "2021-03-28",
  "2022-03-27"
)
rows.dup <- input_df[which((input_df$hour == 3) & input_df$day %in% hour_loss_days),]
rows.dup.names <- rownames(rows.dup)
rows.dup.names.new <- ymd_hms(rows.dup.names) - hours(1)
rows.dup.names.new <- format(rows.dup.names.new, "%Y-%m-%d %H:%M:%S")
rownames(rows.dup) <- rows.dup.names.new
input_df <- rbind(input_df, rows.dup)
input_df <- input_df[order(rownames(input_df)),]
input_df[rownames(rows.dup), "hour"] <- 2

load_cols = c("Calabria", "Centre.North", "Centre.South", "North", "Sardinia", "Sicily",
              "South")

input_df[,load_cols] <- lapply(input_df[,load_cols], as.numeric)
head(input_df)

T <- nrow(input_df)
plot(1:T, input_df$Italy, type = "l")

# Operation to duplicate the load at midnight to add it as point of previous day and next day
rows.dup <- input_df[which(input_df$hour == 0),]
rows.dup.names <- rownames(rows.dup)
rows.dup.names.new <- ymd_hms(rows.dup.names) - days(1) + hours(23) + minutes(59) + seconds(59)
rows.dup.names.new <- format(rows.dup.names.new, "%Y-%m-%d %H:%M:%S")
rownames(rows.dup) <- rows.dup.names.new

df <- rbind(input_df, rows.dup)
df <- df[order(rownames(df)),]

cols.to.shift <- c("year", "day", "weekday", "weekofyear", "monthofyear", "daytype")
rows.to.shift <- grep("59$", rownames(df))[-1]

df[rows.to.shift, "hour"] <- 24

for (col in cols.to.shift) {
  for (row in rows.to.shift) {
    df[row, col] <- df[row-1, col]
  }
}
n <- dim(df)[1]
df <- df[2:(n-1),] # Remove first and last line corresponding to previous or posterior years

# Pivot wider
df_fda <- df[,c(load_cols, "day", "daytype", "hour")]
df_fda <- pivot_wider(df_fda, id_cols = "hour", names_from = c(day, daytype), values_from = load_cols)

# HOTFIX: Check that there are no NA except from Calabria data before 2021
df_fda[, colSums(is.na(df_fda)) > 0 & !grepl("Calabria_2018|Calabria_2019|Calabria_2020", colnames(df_fda))]

df_fda <- df_fda[,colSums(is.na(df_fda)) == 0]
df_fda <- as.data.frame(df_fda)
rownames(df_fda) <- as.numeric(df_fda$hour)
df_fda$hour <- NULL
write.csv(t(df_fda), file = 'data/daily_curves_fixed.csv', row.names = TRUE)

# Smoothing ----------------------------------------------------
T <- 24
t <- 0:T


## Scaling -------------------------------------------------------------------------------------
# We have to be careful in the scaling because we want a functional l1 norm and not a vector
# l1 norm
# Function below compute the l1 norm considering the trapezoidal rule
functional.norm <- function(y, h) {
  n <- length(y) - 1
  res <- 0
  for (i in 1:n) {
    res <- res + y[i] + y[i+1]
  }
  return(res * h / 2)
}

#df_fda_scaled <- scale(df_fda)
df_fda_scaled <- apply(df_fda, 2, function(col) col / functional.norm(col, 1)) # L1 normalization
l1_norm <- apply(df_fda, 2, function(col) functional.norm(col, 1))

# Plot using matplot
#matplot(rownames(df_fda_scaled), df_fda_scaled[,], type = "l", col = 1:ncol(df_fda_scaled),
#        lty = 1, lwd = 2, xlab = "X", ylab = "Y",
#        main = "Daily load curves")

#basis <- create.fourier.basis(rangeval=range(1:T), nbasis=nbasis)

# Set parameters
#m <- 5           # spline order 
#degree <- m-1    # spline degree 
nbasis <- c(7, 9, 13, 17)
nbasis <- c(13)

df.fd.list <- list()
for (p in nbasis) {
  #basis <- create.bspline.basis(rangeval=c(0,24), nbasis=p, norder=m)
  basis <- create.fourier.basis(rangeval=c(0,24), nbasis=p)
  df.fd <- Data2fd(y = as.matrix(df_fda_scaled), argvals = t, basisobj = basis)
  functionalPar <- fdPar(fdobj=df.fd, lambda=0)
  df.smooth <- smooth.pos(argvals = t, y = as.matrix(df_fda_scaled), functionalPar)
  df.fd.list[[as.character(p)]] <- df.smooth
}
#plot.fd(df.fd)

colors <- c("red", "green", "purple", "cyan")

show.random.smoothed.curve <- function(df_fda_scaled, nbasis, df.fd.list, unit='random') {
  if (unit == 'random') {
    unit <- sample(colnames(df_fda), size = 1)
  }
  
  plt <- plot_ly(
    x = t,
    y = df_fda_scaled[,unit],
    type = "scatter",
    mode = "markers",
    name = "Original function"
  )
  
  k <- 1
  for (p in nbasis) {
    df.smooth <- df.fd.list[[as.character(p)]]
    df.Wfd <- df.smooth$Wfdobj
    s.values <- exp(eval.fd(seq(0, 24, 0.1), df.Wfd))
    colnames(s.values) <- colnames(df_fda)
    
    plt <- add_trace(
      plt,
      x = seq(0, 24, 0.1),
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
    showlegend = FALSE  # You can customize other layout options as well
  )
  
  print(plt)
}

show.random.smoothed.curve(df_fda_scaled, nbasis, df.fd.list)

# Best number of basis is 13. How do these smoothed curves look?

nbasis <- 13
eval.grid <- seq(0, 24, 0.25)
df.smooth <- df.fd.list[[as.character(nbasis)]]
df.Wfd <- df.smooth$Wfdobj
s.values <- exp(eval.fd(eval.grid, df.Wfd))
colnames(s.values) <- colnames(df_fda)
rownames(s.values) <- eval.grid

# Export the smoothed curves
smoothed.curves <- t(sweep(s.values, 2, l1_norm, "*"))
write.csv(smoothed.curves, file = 'data/2_processed/daily_curves_smoothed_15min.csv', row.names = TRUE)

plot.full.smoothed.curve <- function(df_fda, s.values) {
  unit <- sample(colnames(df_fda), size = 1)
  
  plt <- plot_ly(
    x = eval.grid,
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
fpca <- pca.fd(df.Wfd, nharm=nbasis, centerfns=TRUE)
cumsum(fpca$varprop)

# Plotting the scree plot
plot(cumsum(fpca$varprop), type = "b", xlab = "Number of Components", ylab = "Cumulative Proportion of Variance",
     ylim = c(0, 1), pch = 19, xaxt = "n", main=NULL)
axis(1, at = 1:length(fpca$varprop), labels = 1:length(fpca$varprop)) # Adding xticks
abline(v = 3, col = "red", lty = 2) # Adding a vertical dashed line at 3 components
grid() # Adding gridlines

plt <- plot_ly(
  x = 1:nbasis,
  y = cumsum(fpca$varprop),
  type = "scatter",
  mode = "lines+markers",
  name = sprintf("Smoothed %d basis", nbasis)
) %>% layout(
  yaxis = list(range=c(0,1)),
  xaxis = list(range=c(1,15.1)),
  title = "Cumulative proportion of variance",
  margin = list(t = 50)
  )

plt <- add_trace(
  plt,
  x = 1:nbasis,
  y = cumsum(fpca$varprop),
  type = "scatter",
  mode = "lines+markers",
  name = sprintf("Smoothed %d basis", nbasis)
  #line = list(color = colors[1])
)

# Add a title to the plot
#plt <- plt %>% layout(
#  title = unit,
#  margin = list(t = 50),  # Set the top margin (adjust the value as needed),
#  showlegend = FALSE  # You can customize other layout options as well
#)

print(plt)


#plot(cumsum(fpca$varprop), type='b', xlab='number of components', 
#     ylab='contribution to the total variance', ylim=c(0,1))


## Interpretation of the PCs ------------------------------------------------------------------
par(mfrow=c(1,2))
plot(fpca, nx=1000, pointplot=TRUE, harm=c(1,2,3,4), expand=0, cycle=FALSE)

plot(fpca$harmonics[1,],col=1,ylab='FPC1')
abline(h=0,lty=2)
plot(fpca$harmonics[2,],col=2,ylab='FPC2')
par(mfrow=c(1,1))

# It seems that the data is not so badly approximated in a 3 or 4 dimensional space (85% or 90% of the variance
# for the system with 13 basis functions)

## Scores -------------------------------------------------------------------------------------

df.scores <- fpca$scores
rownames(df.scores) <- colnames(df_fda)
df.scores <- as.data.frame(df.scores)
df.scores[,"month"] <- sapply(rownames(df.scores), function(x) str_extract(x, ".*-(.*)-.*", group=1), USE.NAMES = FALSE)
df.scores[,"region"] <- sapply(rownames(df.scores), function(x) str_extract(x, "(.*)_.*_.*", group=1), USE.NAMES = FALSE)
df.scores[,"daytype"] <- sapply(rownames(df.scores), function(x) str_extract(x, ".*_.*_(.*)", group=1), USE.NAMES = FALSE)

month <- '02'
daytype <- "Working day"
region <- "North"
df.plot <- df.scores[which((df.scores$region == region)),]
#df.plot <- df.scores[which((df.scores$region == region)&(df.scores$daytype == daytype)),]
#df.plot <- df.scores[which((df.scores$month == month)&(df.scores$daytype == daytype)),]
#df.plot <- data.frame(df.scores)

pcs <- c(1,2)
plot_ly(
  x = df.plot[,pcs[1]],
  y = df.plot[,pcs[2]],
  type = "scatter",
  mode = "markers",
  #colors = "PuOr",
  marker = list(size=8),
  color=df.plot[,"daytype"],
  text = rownames(df.plot)
) %>% layout(
  title = sprintf("Scores for region %s", region),
  margin = list(t = 50),
  xaxis = list(title = sprintf("PC %s", pcs[1])),  # Add x-axis label
  yaxis = list(title = sprintf("PC %s", pcs[2])), # Add y-axis label
  showlegend = TRUE
)


pairs(df.plot[, 1:4], pch = 16, col = as.integer(as.factor(df.plot$daytype)), main = "Scatterplot Matrix")
legend("topright", legend = levels(df.scores$region), col = 1:7, pch = 16, title = "Regions")


## Approximation of the data in the projected space -------------------------------------------
pcbasis <- fpca$harmonics[1:4,]

x.smooth <- seq(0, 24, 0.25)

s.values <- exp(eval.fd(x.smooth, df.Wfd))
colnames(s.values) <- colnames(df_fda)

project.unit <- function(unit, x=x.smooth, nb.pc=4) {
  res <- fpca$meanfd
  for (k in 1:nb.pc) {
    res <- res + df.scores[unit,k] * fpca$harmonics[k,]
  }
  res <- exp(eval.fd(x, res))
  return(res)
}

unit <- sample(colnames(df_fda), size = 1)
plot(x=t, y=df_fda_scaled[,unit], lty=1, col='blue', lwd=2, main=unit, xlim = c(0,24))
lines(x=x.smooth, s.values[,unit], lty=1, col='blue', lwd=2)
lines(x=x.smooth, project.unit(unit, nb.pc = 1), col='red', main=unit, lwd=2)
lines(x=x.smooth, project.unit(unit, nb.pc = 2), col='red', main=unit, lwd=2, lty=2)
lines(x=x.smooth, project.unit(unit, nb.pc = 3), col='red', main=unit, lwd=2, lty=3)
lines(x=x.smooth, project.unit(unit, nb.pc = 4), col='red', main=unit, lwd=2, lty=4)
lines(x=x.smooth, project.unit(unit, nb.pc = 6), col='red', main=unit, lwd=2, lty=5)
legend("topleft", legend=c("Original", "Smoothed", "1 PC", "2 PCs", "3 PCs", "4 PCs", "6 PCs"),
       col=c("blue", "blue", "red", "red", "red", "red", "red"),
       lty=c(NA, 1, 1, 2, 3, 4, 5),
       pch = c(1, NA, NA, NA, NA, NA, NA),
       lwd=2,
       title="Legend Title")

# 3 PCs seems really good


## Exporting the PCA reconstruction of the data ------------------------------------------------
export_projected <- function(nb.pc) {
  res <- matrix(nrow = 0, ncol = length(t))
  pb <- progress_bar$new(format = "[:bar] :percent ETA: :eta", total = dim(df_fda)[2])
  for (unit in colnames(df_fda)) {
    projected <- project.unit(unit, x=t, nb.pc=nb.pc)
    res <- rbind(res, as.vector(projected))
    pb$tick()
  }
  rownames(res) <- colnames(df_fda)
  colnames(res) <- t
  write.csv(res, file = sprintf('data/2_processed/daily_curves_reconstructed_%dPCs.csv', nb.pc), row.names = TRUE)
}

export_projected(nb.pc = 3)
