# Revisión EDIT 2021


# Librerías de trabajo ----------------------------------------------------


library(tidyverse)
library(ggpubr)
library(viridis)
library(ggsci)
library(ggthemes)
library(scales) # Nos interesa la funcion percent()
library(summarytools)



# Data Set EDIT 2021 ------------------------------------------------------


setwd('C:\\Users\\USUARIO\\Desktop\\Trabajo\\Iberoamericana\\Investigación\\EDIT')
getwd()
list.files()

EDIT_19_20 <- read.csv("Datos\\EDIT_X_2019_2020.csv",sep = ';')
dim(EDIT_19_20) # 6798 empresas y 729 variables 

colnames(EDIT_19_20)
# Número de orden (identificador)
sum(duplicated(EDIT_19_20$NORDEMP))
anyDuplicated(EDIT_19_20$NORDEMP)

#Numero de empresas investigadas según actividad económica

length(table(EDIT_19_20$CIIU4)) # 55 actividades económicas

sort(unique(EDIT_19_20$CIIU4)) #Códigos de las actividades

# Actividad económica
table(EDIT_19_20$CIIU4)
tabla <- freq(EDIT_19_20$CIIU4,report.nas = F,order = 'freq')
as.data.frame(tabla)

#Prueba barras apiladas------------

windows()

tabla <- EDIT_19_20 %>% group_by(CIIU4) %>%
  summarise(Count = n())%>%
  mutate(Percentage = (Count / sum(Count)) * 100) %>% round(2)

tabla$CIIU4[tabla$CIIU4 == 108] <- 'Alimentos'
tabla$CIIU4[tabla$CIIU4 == 141] <- 'Confección'
tabla$CIIU4[tabla$CIIU4 == 152] <- 'Calzado'
tabla$CIIU4[tabla$CIIU4 == 181] <- 'Impresión'
tabla$CIIU4[tabla$CIIU4 == 222] <- 'Plasticos'
tabla$CIIU4[tabla$CIIU4 == 239] <- 'Minerales (no-metal)'
tabla$CIIU4[tabla$CIIU4 == 251] <- 'Metales (estructuras)'
tabla$CIIU4[tabla$CIIU4 == 259] <- 'Metales (otros)'
tabla$CIIU4[tabla$CIIU4 == 311] <- 'Muebles'

#Gráfico presentación (%)
tabla %>%
  arrange(desc(Percentage)) %>%  # Ordenar de mayor a menor
  slice(1:9) %>% 
  ggplot(aes(y = reorder(CIIU4, Percentage), x = Percentage, fill = factor(Percentage))) +
  geom_bar(stat = "identity") +
  labs(
    y = "Categoría CIUU4",
    x = "Porcentaje de empresas",
    fill = "",
    title = 'Sectores (CIIU4) con más empresas registradas (50.68%)',
    caption = 'Fuente: Elaboración propia. Datos EDIT (2019-2020)'
  ) +
  theme_light() +
  scale_fill_brewer(palette = "Oranges")+
  geom_text(aes(label = Percentage), vjust = .5)

# TIPOLOGÍA -----------------
# DANE las clasifica en 4 tipos en función de los resultados
#Innovadoras en sentido estricto (11)
# Innovadoras en sentido amplio (1561)
# Potencialmente innovadoras (278)
# No innovadoras (5470)

round(prop.table(table(EDIT_19_20$TIPOLO)) * 100,digits = 1)

levels(as.factor(EDIT_19_20$TIPOLO))

#Aparece la categoría "INTENC" que, siguiendo el boletín
#debe corresponder a "NOINNO"
EDIT_19_20$TIPOLO[EDIT_19_20$TIPOLO == "INTENC"] <- "NOINNO"

# Gráfico de barras: tipología

EDIT_19_20 %>% group_by(TIPOLO) %>% 
  count() %>% rename(frecuencia = n) %>% 
  mutate(relativa = 100*(frecuencia / nrow(EDIT_19_20))) %>% 
  arrange(relativa) %>% 
  ggplot(aes(x = reorder(TIPOLO, relativa),y=relativa, fill = TIPOLO))+geom_col()+
  theme_light()+scale_fill_jama()+
  labs(x = "Tipología de innovación",
       y = "Frecuencia relativa",
       title = "Tipología de innovación según el DANE (2019-2020)",
       caption = 'Fuente: Elaboración propia. Datos EDIT X 2021',
       fill = 'Tipología')+
  scale_x_discrete(labels = c("Estrictos","Potenciales",
                              "Amplios","No innovadores"))+
  scale_y_continuous(labels = percent_format(scale = 1))+
  geom_text(aes(label = scales::percent(relativa / 100),
                y = relativa), vjust = -0.5, size = 4)



# CIIU y tipologia --------------------------------------------------------


table(EDIT_19_20$TIPOLO)
table(EDIT_19_20$CIIU4)
unique(EDIT_19_20$CIIU4)

df <- data.frame('CIIU4' = EDIT_19_20$CIIU4,
                 'Tipologia' = EDIT_19_20$TIPOLO)

contin <- prop.table(table(df$Tipologia,df$CIIU4),margin = 2)

contin <- as.data.frame(contin)
colnames(contin) <- c('Tipologia','CIIU4','Freq')

contin %>%  sample_n(5) #Vista aleatoria de mi df


#Sectores mas innovadores en sentido amplio
df <- contin %>% filter(Tipologia == 'AMPLIA') %>% 
  arrange(desc(Freq)) %>% slice(1:10)

unique(df$CIIU4)
df$CIIU4 <- as.character(df$CIIU4)
df$CIIU4 <- as.factor(df$CIIU4) #Ahora si se ven sólo 10 niveles

contin <- contin[contin$CIIU4%in%df$CIIU4,]
contin$CIIU4 <- as.character(contin$CIIU4)
contin$CIIU4 <- as.factor(contin$CIIU4)
levels(contin$CIIU4)

nom_levs <- c('Carnes','Frutas, legumbres(otros)','Aceites','Lácteos',
              'Alimentos animales', 'Plaguicidas','Jabones y detergentes',
              'Farmaceuticos','Informáticos','Vehiculos')
          
levels(contin$CIIU4) <- nom_levs #let's Changes levels

ggplot(data = contin, aes(y = CIIU4, x = Freq, fill = Tipologia)) +
  geom_bar(stat = "identity") +
  labs(
    y = "CIIU4",
    x = "Porcentaje",
    fill = "Tipologia",
    title = "Sectores CIIU4 con mas innovación en sentido amplio",
    caption = 'Fuente: Elaboración propia. Datos EDIT X 2021'
  ) + scale_x_continuous(labels = scales::percent) +  # Escala en porcentaje
  theme_light()+
  geom_text(aes(label = sprintf("%.0f%%", Freq*100)),
            position = position_stack(vjust = 0.5),
            color = "white", size = 4)+
  scale_fill_jama()


# Sectores mas innovadores en sentido estricto

table(EDIT_19_20$TIPOLO) # Solamente hay 11 empresas innovadoras



# Innovación de bienes y servicios ----------------------------------------


#Bienes o servicios nuevos únicamente para su empresa
#(Ya existían en el mercado nacional y/o en el internacional)

table(EDIT_19_20$I1R1C1N) 
table(EDIT_19_20$I1R1C2N) # Cantidad

#Bienes o servicios nuevos en el mercado nacional (Ya
#existían en el mercado internacional).

table(EDIT_19_20$I1R3C1N)
table(EDIT_19_20$I1R2C2N) # Cantidad

#Bienes o servicios nuevos en el mercado internacional.
table(EDIT_19_20$I1R3C1N)
table(EDIT_19_20$I1R3C2N) # Cantidad

# Validamos que I1R4C2N es la suma de las innovaciones ----- 

EDIT_19_20 %>% select(I1R1C2N,I1R2C2N,I1R3C2N,I1R4C2N) %>% 
  mutate(
    I1R1C2N = ifelse(is.na(I1R1C2N) | I1R1C2N == "NOINNO", 0, as.numeric(I1R1C2N)),
    I1R2C2N = ifelse(is.na(I1R2C2N) | I1R2C2N == "NOINNO", 0, as.numeric(I1R2C2N)),
    I1R3C2N = ifelse(is.na(I1R3C2N) | I1R3C2N == "NOINNO", 0, as.numeric(I1R3C2N)),
    I1R4C2N = ifelse(is.na(I1R4C2N) | I1R4C2N == "NOINNO", 0, as.numeric(I1R4C2N)),
    innovaciones = I1R1C2N + I1R2C2N + I1R3C2N
  ) -> eje    
  
table(eje$innovaciones)==table(eje$I1R4C2N)  

# Terminamos validación 

#############################################

# Número total de innovaciones de bienes o servicios nuevos
table(EDIT_19_20$I1R4C2N)
#############################################


# Innovación en bienes y servicios mejorados ------------------------------

# Bienes o servicios significativamente mejorados para su empresa

table(EDIT_19_20$I1R1C1M)
table(EDIT_19_20$I1R1C2M) #Cantidad

#Bienes o servicios significativamente mejorados en el mercado nacional 

table(EDIT_19_20$I1R2C1M)
table(EDIT_19_20$I1R2C2M) #Cantidad

# Bienes o servicios significativamente mejorados en el mercado internacional.

table(EDIT_19_20$I1R3C1M)
table(EDIT_19_20$I1R3C2M) #Cantidad

#############################################
# Total innovaciones de bienes o servicios significativamente mejorados
table(EDIT_19_20$I1R4C2M)
#############################################

# Innovación en métodos de producción -------------------------------------

#mejorados métodos de producción, distribución, entrega
table(EDIT_19_20$I1R4C1)
table(EDIT_19_20$I1R4C2) #Cantidad

#métodos organizativos implementados en el funcionamiento interno de la
#empresa

table(EDIT_19_20$I1R5C1)
table(EDIT_19_20$I1R5C2) #Cantidad

#nuevas técnicas de comercialización en su empresa

table(EDIT_19_20$I1R6C1)
table(EDIT_19_20$I1R6C2) #Cantidad



# Valoración del impacto de la innovación ---------------------------------


#Mejora en la calidad de los bienes 
#Nivel de importancia

table(EDIT_19_20$I2R1C1)#Alta(1),media(2),nula(3)

# Ampliación en la gama de bienes o servicios
table(EDIT_19_20$I2R2C1) #Alta(1),media(2),nula(3)

#Ha mantenido su participación en el mercado geográfico de su empresa

table(EDIT_19_20$I2R3C1)

# Ha ingresado a un mercado geográfico nuevo

table(EDIT_19_20$I2R4C1) 

# Aumento de la productividad

table(EDIT_19_20$I2R5C1)

# Reducción de los costos laborales

table(EDIT_19_20$I2R6C1) 

#Reducción en el uso de materias primas o insumos.

table(EDIT_19_20$I2R7C1)

# Mejora en el cumplimiento de regulaciones, normas y reglamentos técnicos.

table(EDIT_19_20$I2R13C1) 




# Porcentaje de impacto ---------------------------------------------------
# pag. 25
table(EDIT_19_20$I4R1C1)#ya exisitian a nivel nal (nal)
table(EDIT_19_20$I4R1C2)#ya exisitian a nivel nal (expor)
table(EDIT_19_20$I4R2C1)#Productos ya existentes extranje (nal)
table(EDIT_19_20$I4R2C2)#Productos ya existentes extranje (exporta)
table(EDIT_19_20$I4R3C1)#Mejoras (ventas nacionales)
table(EDIT_19_20$I4R3C2)# Mejoras (exportaciones)
table(EDIT_19_20$I4R4C1)# productos no innovadores (ventas)
table(EDIT_19_20$I4R4C2)# productos no innovadores (exportaciones)


# Resultados económicos ---------------------------------------------------

options(scipen=999)

#Ventas nacionales totales (miles de pesos corrientes)

EDIT_19_20$I3R1C1 #2019
EDIT_19_20$I3R2C1 #2020

df_macro <- as.dataframe(cbind(EDIT_19_20$I3R1C1,2019))
colnames(df_macro) <- c('Ventas','Anio') 

df_macro <- as.data.frame(rbind(df_macro,cbind(EDIT_19_20$I3R2C1,2020)))

ggplot(data = df_macro[-c(EDIT_19_20$I3R1C1 == max(EDIT_19_20$I3R1C1)),],aes(x = Ventas, fill= as.factor(Anio)))+
  geom_density(alpha = .4)+theme_light() 


quantile(df_macro$Ventas)

df_macro[which(df_macro$Ventas == max(df_macro$Ventas)),]

eje <- EDIT_19_20[which(EDIT_19_20$I3R1C1 == max(EDIT_19_20$I3R1C1)),
           c('NORDEMP','CIIU4','TIPOLO','I3R1C1','I3R2C1')]

eje$prueba <- eje$I3R2C1 - eje$I3R1C1  
eje




# Variables explicativas del modelo ---------------------------------------

# 1. Tamaño de la empresa [TE](personal ocupado)

sum(is.na(EDIT_19_20$IV1R11C1)) #0
sum(is.na(EDIT_19_20$IV1R11C1)) #0

hist(EDIT_19_20$IV1R11C1)# Personal ocupado 2019

hist(EDIT_19_20$IV1R11C2) #Personal ocupado 2020

# 2. Cualificación del persona (QS, qualifies staff) 
# Doctorados, maestria, especialización

EDIT_19_20$QS_2019 <- 100*(EDIT_19_20$IV1R1C1+EDIT_19_20$IV1R2C1+EDIT_19_20$IV1R3C1)/(EDIT_19_20$IV1R11C1)

EDIT_19_20$QS_2020 <- 100*(EDIT_19_20$IV1R1C2+EDIT_19_20$IV1R2C2+EDIT_19_20$IV1R3C2)/(EDIT_19_20$IV1R11C2)


EDIT_19_20[is.na(EDIT_19_20$QS_2020), 
           c('IV1R11C1','IV1R11C2','QS_2019','QS_2020')] %>% 
  sample_n(20)

sum(is.na(EDIT_19_20$QS_2019))
sum(is.na(EDIT_19_20$QS_2020)) #114 datos perdidos, no trabajadores

sum(EDIT_19_20$IV1R11C1)-sum(EDIT_19_20$IV1R11C2)#perdida de empleo(46054)

#
  
# Exportaciones totales (miles de pesos corrientes)

EDIT_19_20$I3R1C2
EDIT_19_20$I3R2C2

# Ventas nacionales totales 2020 (Miles de pesos corrientes)

table(EDIT_19_20$I3R2C1)

ventas20 <- hist(as.numeric(EDIT_19_20$I3R2C1))
ventas20 <- data.frame(cbind(ventas20$breaks,ventas20$counts))
colnames(ventas20) <- c('Intervalo','Frecuencia')
ventas20

# Exportaciones totales 2020

hist(EDIT_19_20$I3R2C2) 



sum(is.na(EDIT_19_20$II1R9C1)) #2017 [5679 no reportan]


df_model21 <-EDIT_19_20[!is.na(EDIT_19_20$II1R10C2),]
table(df_model21$CIIU4)

df_model21 <- df_model21[,c('NORDEMP','TIPOLO','CIIU4','IV1R11C2','II1R1C2','II1R2C2','II1R3C2','II1R4C2','II1R5C2','II1R6C2',
                        'II1R7C2','II1R8C2','II1R9C2','III1R3C2')]

nombres <- c('NORDEMP','TIPOLO','CIIU4','Personal_18','I+D_inversion','I+D_adquisicion','Maquinaria_adq',
             'Info_teleco','Mercadotecnia','Transfer_tecno','Asis_tecnica','Inge_industrial',
             'Formacion','Rec_publicos')
colnames(df_model21) <- nombres

table(df_model21$TIPOLO)

table(EDIT_19_20$TIPOLO)

