rm(list=ls())
library('plot3D')
library('mlbench')
library('roccv')


p <- mlbench.spirals(1000, cycles=1, sd= 0.05)
plot(p, main = "Dados de Entrada")
class1 <- p$x[which(p$classes == 1), ]
class2 <- p$x[which(p$classes == 2), ]

numFolds <- 10
folds_c1 <- randomly_assign(dim(class1)[1],numFolds)
folds_c2 <- randomly_assign(dim(class2)[1],numFolds)

verossimilhancasC1 <- matrix(0,100,2)
verossimilhancasC2 <- matrix(0,100,2)

acc <- vector()

for(i in 1:numFolds) {
  cross1_test <- class1[which(folds_c1 == i),]
  cross2_test <- class2[which(folds_c2 == i),]
  cross1_train <- class1[which(folds_c1 != i),]
  cross2_train <- class2[which(folds_c2 != i),]
  test <- rbind(cross1_test, cross2_test)
  gabarito <- rbind(array(1, c(nrow(cross1_test),1)), array(2, c(nrow(cross2_test),1)))
  
  n_test <- nrow(test)
  y_hat <- vector()
  n1 <- nrow(cross1_train)
  n2 <- nrow(cross2_train)
  N_total <- n1 + n2
  
  h1 <- 1.06*sd(cross1_train)*N_total**(-1/5)
  h2 <- 1.06*sd(cross2_train)*N_total**(-1/5)
  
  
  for (j in 1:n_test) {
    kernel_C1 <- kernel(test[j, ],cross1_train, h1)
    kernel_C2 <- kernel(test[j, ],cross2_train, h2)
    
    pc1 <- probKDE(h1,N_total,kernel_C1)
    pc2 <- probKDE(h2,N_total,kernel_C2)
    
    verossimilhancasC1[j, ] <- c(pc1,0)  
    verossimilhancasC2[j, ] <- c(0,pc2)
    
    y_hat[j] <- bayesC((n1/N_total)*pc1, (n2/N_total)*pc2)
  }
  acc[i]<- (1 - calc_error(gabarito, y_hat))*100
}
print(sd(acc))
print(mean(acc))

plot(verossimilhancasC2, col="red", xlab="P(x|C1)", ylab="P(x|C2)" ,xlim=c(0, max(verossimilhancasC1)), ylim=c(0,max(verossimilhancasC2)))
par(new = T)
plot(verossimilhancasC1, col="blue", xlab = "" , ylab="" ,yaxt = "n" , xaxt ="n", xlim=c(0, max(verossimilhancasC1)), ylim=c(0,max(verossimilhancasC2)))