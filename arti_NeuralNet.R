# NeuralNetTools: Visualization and Analysis Tools for Neural Networks

# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6262849/#R19

# Olden y Jackson (2002)describen el diagrama de interpretación neuronal 
# (NID) para trazar ( Özesmi y Özesmi 1999 ), el algoritmo de Garson para
# la importancia de la variable ( Garson 1991 ) y el método de perfil para
# el análisis de sensibilidad

# NeuralNetTools incluye cuatro funciones principales que fueron 
# desarrolladas siguiendo técnicas similares en Olden y Jackson (2002)
# y referencias en el mismo:
# plotnet(),  garson(), olden, lekprofile() for a sensitivity analysis

#Un enfoque común para el preprocesamiento de datos es normalizar
# las variables de entrada y estandarizar las variables de respuesta 
# ( Lek y Guégan 2000 ; Olden y Jackson 2002 ).

#Data set
library(NeuralNetTools)

View(NeuralNetTools::neuraldat)
data(neuraldat)

set.seed(123) #se mantienene los datos aleatorios creados
library(RSNNS)

x <- neuraldat[,c("X1", "X2", "X3")]
y <- neuraldat[, "Y1"]
mod1 <- mlp(x, y, size = 5)
library(neuralnet)
mod2 <- neuralnet(Y1 ~ X1 + X2 + X3, data = neuraldat, hidden = 5)
library(nnet)
mod3 <- nnet(Y1 ~ X1 + X2 + X3, data = neuraldat, size = 5)

ls()  # Lista todos los objetos en tu entorno de trabajo



# Visualizing neural networks ---------------------------------------------



pruneFuncParams <- list(max_pr_error_increase = 10.0,
                        pr_accepted_error = 1.0, no_of_pr_retrain_cycles = 1000,
                        min_error_to_stop = 0.01, init_matrix_value = 1e-6,
                        input_pruning = TRUE, hidden_pruning = TRUE)

mod <- mlp(x, y, size = 5, pruneFunc = "OptimalBrainSurgeon",
           pruneFuncParams = pruneFuncParams)

# Red final tras ser podada
plotnet(mod, rel_rsc = c(3, 8))
#Red final pero con las conexiones podadas

plotnet(mod, prune_col = "lightblue", rel_rsc = c(3, 8))
?plotnet()
# Evaluación de la importancia de la variable -----------------------------

# Olden’s connection weights algorithm
# Algoritmo de ponderación de conexiones de Olden
library(ggplot2)

#Estos gráficos pueden ser ajustados con ggplot2
garson(mod1)+theme_light()+ggtitle('Algoritmo de Garson')
olden(mod1)
garson(mod2)
olden(mod2)
garson(mod3)
olden(mod3)


# Análisis de sensibilidad ------------------------------------------------

lekprofile(mod3)
lekprofile(mod3, group_show = TRUE)
lekprofile(mod3, group_vals = 6)
lekprofile(mod3, group_vals = 6, group_show = TRUE)


# Ejemplo aplicado --------------------------------------------------------

library(nycflights13)
library(dplyr)

#Preprocesamiento inicial de datos para normalizar las entradas,
#estandarizar la respuesta y evaluar la influencia de los
#valores atípicos.

filter(flights, month == 12 & carrier == 'UA') %>% 
  select(arr_delay, dep_delay, dep_time, arr_time, air_time,
         distance) %>% mutate_each(funs(scale), -arr_delay) %>% 
  mutate_each(funs(as.numeric), -arr_delay) %>%
  mutate(arr_delay = scales::rescale(arr_delay, to = c(0, 1))) %>%
  data.frame -> tomod

tomod
#multilayer perceptron (MLP) 
#La forma más popular de red neuronal es el perceptrón 
#multicapa de retroalimentación (MLP) entrenado utilizando 
#el algoritmo de retropropagación ( Rumelhart et al. 1986 ). 

library(nnet)
mod <- nnet(arr_delay ~., size = 5, linout = TRUE, data = tomod,
               trace = FALSE)

library(NeuralNetTools)
windows()
plotnet(mod) #Estructura del modelo.
#Acá se puede evaluar los pesos negativos, que se leen como
#una relación opuesta, v.gr., dep_delay y dep_time con arr_delay
garson(mod) 
olden(mod) 
a<- lekprofile(mod,group_vals = 5) 
a <- lekprofile(mod, group_vals = 5, group_show = TRUE)
a 
