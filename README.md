# Projeto_7_Sistema_de_Recomendacao
**Sistema de recomendação de produtos para consumidores**

## DESCRIÇÃO:

Utilizando o TensorFlow desenvolvi um modelo que foi treinado, a partir de uma base de dados, e que ao final lista os 07 (sete) melhores produtos que podem ser recomendados ao consumidor.<br>
No modelo estão os 07 (sete) melhores produtos que podem ser recomendados para o consumidor de nome 'Darrin Van Huff'.<br>
Porém, como eu criei uma função para gerar essa tabela final com as recomendações para o consumidor, o mesmo pode ser substituído por qualquer outro. Assim podemos ter as melhores recomendações para qualquer cliente que desejarmos.<br><br>

No arquivo inicial, foi feita uma análise exploratória e, com isso, vi a necessidade de algumas correções na base:<br>
- Eliminação dos valores nulos;
- Correção de alguns tipo (pypes) de dados;
- Foi feito o downcast dos dados;
- Ao final os dados foram salvos na extensão parquet.<br><br>

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
A função 'recomendar_produtos', que se encontra ao final no notebook, permite que novos clientes, ou seja, clientes que não façam parte da base de dados original, também tenham suas recomendações dos 07 (sete) melhores produtos contempladas.
