\# Projeto Final - Física Computacional

\## Programa de Pós Graduação em Física - DF UFPE



Neste projeto, iremos imergir no mundo dos sólitons e dos defeitos topológicos. O projeto será dividido em duas partes:



* Classificação dos resultados possíveis das colisões;
* Tentativa de prever o resultado de uma reflexão no modelo $\\lambda \\phi^4$.



\# Metodologia

No primeiro caso, faremos uso do método PCA seguido do K-Means, utilizando uma quantidade de grupos igual a 3, na separação, pois temos a possibilidade de ter uma reflexão, uma aniquilação e o que chamamos de "rebote" (colidem duas vezes e se separam). O método realizou isso de forma eficaz.



E a segunda técnica, é o treinamento de uma rede neural, o treinamento de um solver para a solução das equações de campo. Nós desenvolvemos um método que treina uma rede neural num ansartz hiperbólico para a colisão e ela aprendeu bem os padrões. O ansartz hiperbólico consegue capturar os efeitos da reflexão.



\# Futuro



No futuro, pretendo trabalhar e implementar a previsão da reflexão dos dois outros resultados. Unindo os dois métodos, conseguimos reconhecer bem onde estão as chamadas janelas de ressonância e, a partir do momento que sabemos reconhecer, podemos utilizar o ansartz correto para então conseguir prever os resultados das colisões. Ainda preciso pensar a respeito dos ansartz a serem utilizados para a aniquilação (mas imagino que um ansartz exponencial decrescente seja o suficiente) e também os rebotes (esse será mais difícil, precisamos pensar numa função intermediária entre aniquilação e reflexão).



Vale ressaltar que esse material possui um poder de publicação, é uma área não tão explorada no seguimento dos defeitos topológicos. Pretendo tratar esses tópicos em futuras reuniões de pesquisa com meus orientadores.

