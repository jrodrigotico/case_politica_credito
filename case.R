# # pacotes
# pacotes <- c("tidyverse", "knitr", "kableExtra", "tm", "SnowballC",
#             "text2vec", "LDAvis", "caret", "e1071", "PROC", 
#             "car","rgl","reshape2", 'readxl')

# if(sum(!(pacotes %in% installed.packages())) != 0) {
#     for(i in which(!(pacotes %in% installed.packages()))) {
#         install.packages(pacotes[i], dependencies = T)
#         if(!require(pacotes[i], character = T)) {
#     break}}} else {
#     sapply(pacotes, require, character = T)
# }

library(tidyverse)
library(ggplot2)
library(readxl)
library(reshape2)
library(corrplot)
library(caTools)
library(caret)
library(car)
library(PROC)



# bases de dados
df <- read_xlsx("dados_case.xlsx", sheet = 'dados')
df2 <- df
df3 <- df

# Taxa de inadimplencia atual, vendo apenas a variavel Atraso_apos_6meses
total_linhas <- nrow(df2)
total_atraso <- sum(df2$Atraso_apos_6meses == 1)
tx_atual <- total_atraso / total_linhas

print(paste("Taxa de inadimplência hoje: ", tx_atual * 100, "%"))


# Taxa de inadimplencia usando Score_de_credito
tx_df2_600 <- sum(df2$Score_de_credito > 600 & df2$Atraso_apos_6meses == 1) / sum(df2$Score_de_credito > 600)
tx_df2_500 <- sum(df2$Score_de_credito <= 500 & df2$Atraso_apos_6meses == 1) / sum(df2$Score_de_credito <= 500)

sprintf("Taxa de inadimplência com score acima de 600: %.2f%%", tx_df2_600 * 100)
sprintf("Taxa de inadimplência com score 500 ou menor: %.2f%%", tx_df2_500 * 100)

# Nova política de credito
## dados ausentes
for (i in colnames(df2)) {
    resultado <- sum(is.na(df2[[i]]))  # zero dados ausentes em todas as colunas
    print(paste("NA na coluna:", i))
    print(resultado)
}

## correlação (multicolinearidade)
matriz_corr <- cor(df2[, sapply(df2, is.numeric)])
print(matriz_corr)

## distribuicao



## Padronizacao dos dados
df2$renda <- scale(df2$renda)
df2$Score_de_credito <- scale(df2$Score_de_credito)
head(df2)

## base treino e teste
set.seed(123)
train <- caret::createDataPartition(df2$Atraso_apos_6meses, p = 0.7, list = FALSE)
df_treino <- df2[train, ]
df_teste <- df2[-train, ] # 30% para base de teste


## modelo
modelo <- glm(Atraso_apos_6meses ~ idade + genero + renda + Score_de_credito + # como a coluna 'Contratos' só tinha valor 1, ela n acrescenta nenhuma informação relevante para o modelo
                localizacao + nivel_escolaridade + contas_bancarias,
                data = df_treino,  
                family = binomial)  
summary(modelo) # se p-valor menor que 0.05, a variável é estatisticamente significativa

## validacao
predicoes_prob <- predict(modelo, newdata = df_teste, type = "response")
predicoes_classe <- ifelse(predicoes_prob >= 0.7, 1, 0) # corte de 0.5, ponto de inflexão

matriz_confusao <- confusionMatrix(as.factor(predicoes_classe), as.factor(df_teste$Atraso_apos_6meses), positive = "1")
matriz_confusao







# ## Testando inserir novos clientes
# novo_cliente <- data.frame(
#     idade = 30,  
#     genero = "Masculino",  
#     renda = 4000,   
#     Score_de_credito = 650,  
#     localizacao = "Sul", 
#     nivel_escolaridade = "Superior", 
#     contas_bancarias = 3 
# )

# # padronizando renda e score_credito do novo cliente

# novo_cliente$renda <- as.numeric(novo_cliente$renda)
# novo_cliente$Score_de_credito <- as.numeric(novo_cliente$Score_de_credito)
# novo_cliente$genero <- factor(novo_cliente$genero, levels = c("Masculino", "Feminino"))
# novo_cliente$localizacao <- factor(novo_cliente$localizacao, levels = c("Norte", "Sul", "Leste", "Oeste"))
# novo_cliente$nivel_escolaridade <- factor(novo_cliente$nivel_escolaridade, levels = c("Médio", "Superior", "Pós-graduação"))

# # Calcular a média e o desvio padrão para 'renda' e 'Score_de_credito' no conjunto de treino
# media_renda <- mean(df_treino$renda)
# desvio_renda <- sd(df_treino$renda)

# media_score <- mean(df_treino$Score_de_credito)
# desvio_score <- sd(df_treino$Score_de_credito)

# # Padronizar as variáveis 'renda' e 'Score_de_credito' do novo cliente com base no treino
# novo_cliente$renda <- (novo_cliente$renda - media_renda) / desvio_renda
# novo_cliente$Score_de_credito <- (novo_cliente$Score_de_credito - media_score) / desvio_score

# # previsão para o novo cliente
# probabilidade <- predict(modelo, newdata = novo_cliente, type = "response")
# classe <- ifelse(probabilidade >= 0.5, 1, 0)

# # Exibir o resultado
# cat("Probabilidade de inadimplência do novo cliente:", probabilidade, "\n")
# cat("Classe do novo cliente (1 = inadimplente, 0 = não inadimplente):", classe, "\n")


