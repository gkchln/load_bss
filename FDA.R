library(fda)
library(plotly)
library(stringr)

file_path <- 'data/daily_curves.csv'
df_fda <- t(read.csv(file_path, row.names = 1, check.names = FALSE))

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
#nbasis <- c(7, 9, 13, 17, 23)
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

# Saving the fd list
#save(df.fd.list, file = "data/2_processed/RData/df_fd_list.RData")
#load("data/2_processed/RData/df_fd_list.RData")

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
    showlegend = FALSE,  # You can customize other layout options as well,
    yaxis = list(range = c(0, 0.06))
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
#write.csv(smoothed.curves, file = 'data/2_processed/daily_curves_Italy_smoothed_15min.csv',
#          row.names = TRUE)

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
    showlegend = FALSE,  # You can customize other layout options as well
    yaxis = list(range = c(0, 0.06))
  )
  print(plt)
}

plot.full.smoothed.curve(df_fda, s.values)

# FPCA ----------------------------------------------------------------------------------------
## Cumulative proportion of variance ----------------------------------------------------------
fpca <- pca.fd(df.Wfd, nharm=nbasis, centerfns=TRUE)
cumsum(fpca$varprop)

# Plotting the scree plot
par(mfrow=c(1,1))
plot(cumsum(fpca$varprop), type = "b", xlab = "Number of Components", ylab = "Cumulative Proportion of Variance",
     ylim = c(0, 1), pch = 19, xaxt = "n", main=NULL)
axis(1, at = 1:length(fpca$varprop), labels = 1:length(fpca$varprop)) # Adding xticks
abline(v = 4, col = "red", lty = 2) # Adding a vertical dashed line at elbow
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

# Retrieve and export the "profiles" associated with the different PCs
h = 0.25
x <- seq(0, 24, h)
W.mean <- eval.fd(x, fpca$meanfd)

profiles <- matrix(nrow = 0, ncol = length(x))
rowlabels <- c()
par(mfrow=c(2, 2))
for (k in 1:4) {
  profile.pos <- exp(W.mean + eval.fd(x, fpca$harmonics[k,] * 5 * sqrt(fpca$values[k])))
  print(functional.norm(profile.pos, h)) # They have norm close to 1
  profile.neg <- exp(W.mean - eval.fd(x, fpca$harmonics[k,] * 5 * sqrt(fpca$values[k])))
  print(functional.norm(profile.neg, h)) # They have norm close to 1
  # Plot the profiles
  plot(exp(W.mean), x=x, type='l', lwd=2, ylab='Normalised Load', ylim=c(0, 0.07), xlab='Hour',
       main=sprintf('FPC %d (%.1f%%)', k, 100*fpca$varprop[k]))
  lines(profile.pos, x=x, col=3)
  lines(profile.neg, x=x, col=2)
  # Add them to the export
  profiles <- rbind(profiles, as.vector(profile.pos))
  profiles <- rbind(profiles, as.vector(profile.neg))
  rowlabels <- c(rowlabels, c(sprintf('prof_%d_pos', k), sprintf('prof_%d_neg', k)))
  #Sys.sleep(1)
}
rownames(profiles) <- rowlabels
colnames(profiles) <- x

#write.csv(profiles, file = sprintf('data/2_processed/regional/FPCA_profiles_%dmin.csv', h * 60), row.names = TRUE)

# It seems that the data is not so badly approximated in a 3 or 4 dimensional space (85% or 90% of the variance
# for the system with 13 basis functions)

## Scores -------------------------------------------------------------------------------------

df.scores <- fpca$scores
rownames(df.scores) <- colnames(df_fda)
df.scores <- as.data.frame(df.scores)
df.scores[,"month"] <- sapply(rownames(df.scores), function(x) str_extract(x, ".*-(.*)-.*", group=1), USE.NAMES = FALSE)
df.scores[,"region"] <- sapply(rownames(df.scores), function(x) str_extract(x, "^(.*)_.*_.*", group=1), USE.NAMES = FALSE)
df.scores[,"daytype"] <- sapply(rownames(df.scores), function(x) str_extract(x, ".*_.*_(.*)$", group=1), USE.NAMES = FALSE)

# HOTFIX: add the season to df.scores
month_to_season <- c("Winter", "Winter", "Winter", "Spring", "Spring", "Spring",
                     "Summer", "Summer", "Summer", "Fall", "Fall", "Fall")
df.scores[,"season"] <- month_to_season[as.numeric(df.scores$month)]


#month <- '02'
#daytype <- "Working day"
region <- paste(zones, collapse = '_')
df.plot <- df.scores
#df.plot <- df.scores[which((df.scores$region == region)),]
#df.plot <- df.scores[which((df.scores$region == region)&(df.scores$daytype == daytype)),]
#df.plot <- df.scores[which((df.scores$month == month)&(df.scores$daytype == daytype)),]
#df.plot <- data.frame(df.scores)

pcs <- c(5, 6)
plot_ly(
  x = df.plot[,pcs[1]],
  y = df.plot[,pcs[2]],
  type = "scatter",
  mode = "markers",
  #colors = "PuOr",
  marker = list(size=8),
  color=df.plot[,"month"],
  text = rownames(df.plot)
) %>% layout(
  #title = sprintf("Scores for region %s", region),
  margin = list(t = 50),
  xaxis = list(title = sprintf("PC %s", pcs[1])),  # Add x-axis label
  yaxis = list(title = sprintf("PC %s", pcs[2])), # Add y-axis label
  showlegend = TRUE
)

#pairs(df.plot[, 1:4], pch = 16, col = as.integer(as.factor(df.plot$daytype)), main = "Scatterplot Matrix")
#legend("topright", legend = levels(df.scores$region), col = 1:7, pch = 16, title = "Regions")


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

unit <- sample(rownames(df.scores), size = 1)
par(mfrow=c(1,1))
plot(x=t, y=df_fda_scaled[,unit], lty=1, col='blue', lwd=2, main=unit, xlim = c(0,24))
lines(x=x.smooth, s.values[,unit], lty=1, col='blue', lwd=2)
#lines(x=x.smooth, project.unit(unit, nb.pc = 1), col='red', main=unit, lwd=2)
lines(x=x.smooth, project.unit(unit, nb.pc = 2), col='red', main=unit, lwd=2, lty=2)
lines(x=x.smooth, project.unit(unit, nb.pc = 3), col='red', main=unit, lwd=2, lty=3)
lines(x=x.smooth, project.unit(unit, nb.pc = 4), col='red', main=unit, lwd=2, lty=4)
#lines(x=x.smooth, project.unit(unit, nb.pc = 6), col='red', main=unit, lwd=2, lty=5)
legend("topleft", legend=c("Original", "Smoothed", "1 PC", "2 PCs", "3 PCs", "4 PCs", "6 PCs"),
       col=c("blue", "blue", "red", "red", "red", "red", "red"),
       lty=c(NA, 1, 1, 2, 3, 4, 5),
       pch = c(1, NA, NA, NA, NA, NA, NA),
       lwd=2,
       title="Legend Title")

# 4 PCs seems really good


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
