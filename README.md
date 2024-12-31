# Sistema de Recomendações (Sam's Club - Walmart)
**Sistema de recomendações de produtos para consumidores do Sam's Club - Walmart**

## DESCRIÇÃO:

Este modelo implementa um sistema de recomendação baseado em técnicas de Deep Learning, utilizando TensorFlow (Keras) para treinar um modelo com uma base de dados. O sistema gera, ao final, uma lista dos 07 (sete) melhores produtos recomendados para o consumidor.
No exemplo apresentado, os 07 (sete) produtos recomendados são para o cliente 'Darrin Van Huff'. No entanto, desenvolvi uma função que possibilita a criação dessa tabela final de recomendações para qualquer consumidor. Assim, é possível obter as melhores recomendações personalizadas para qualquer cliente desejado.<br><br>

No arquivo inicial, foi feita uma análise exploratória e, com isso, vi a necessidade de algumas correções na base:<br>
- Eliminação dos valores nulos;
- Correção de alguns tipo (types) de dados;
- Foi feito o downcast dos dados;
- Ao final os dados foram salvos na extensão parquet.<br><br>

## ALGUMAS TECNOLOGIAS UTILIZADAS:
- Python;
- Pandas;
- Numpy;
- TensorFlow (Keras);
- SHAP.<br><br>

O arquivos final inicia importando a base no formato parquet e os passos que se seguem são os seguintes:
- Filtrar colunas relevantes;
- Codificar os nomes dos clientes e IDs dos produtos;
- Normalizar as vendas (utilizei o MinMaxScaler);
- Criar um conjunto de dados TensorFlow;
- Definir dimensões dos embeddings;
- Criando os embeddings;
- Definição do modelo;
- Concatenar todas as informações;
- Camadas densas para aprender padrões;
- Treinar o modelo;
- Obter o ID da cliente "Irene Maddox" (Utilizei essa cliente para o modelo criar recomendações);
- Criar recomendações para Irene;
- Obter os 7 melhores produtos recomendados;
- Criar um DataFrame para exibir as recomendações;
- Função para determinar as melhores 7 recomendações para o cliente escolhido;
- Utilizando a função 'recomendar_produtos';
- Utilizando a função (Nesse momento utilizei o cliente "Darrin Van Huff" como exemplo).<br><br>

### OBSERVAÇÃO:
A função 'recomendar_produtos', desenvolvida por mim e presente no notebook final, possibilita que novos clientes, ou seja, aqueles que não integram a base de dados original, também recebam recomendações dos 07 (sete) melhores produtos.
