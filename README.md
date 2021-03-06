# AlphaGo_Gomoku
This is a small version of AlphaGo Zero written with Keras.
It can play Gomoku like human.
Keras version:2.1.5  



## Model in Detial
### CNN block



* input: raw boards * 4

       1st 1 if current_move else 0
       2nd 1 if opposite_move else 0
       3rd 1 if last_move else 0
       4th all 1 if current_player is on the offensive
       
       
* output: list p for every available position
        value v for confidence of winning
        
* structure:  3*convs + {conv+flatten+dense, conv+flatten+dense+dense}



### MCTS block
A Mento Calo Tree is combined by MCT Nodes
#### MCT Node
A BiDirect Tree Node that contains:
* Parent
* Children
* perior probability p
* action value Q(s,a)
* UCT alogrithm U

It can:<br>
* **select**             choose move by Q + U greedy
* **expand**             expand a leaf node by actions and probabilities
* **update**             N and Q

With the help of MCT Node, a MCTS can:<br>
* **play_a_step**        choose a move greedy until leaf node, expand chilren by CNN blocks
* **simulate**           simulate and update MCT in N times, return all actions and probabilities
* **move**               move with highest probability

### Training 
We collect self play data with dirac noise and shift with randomly rotate.<br>
Train CNN with best policy Pi  and winning rate v generated by MCTS.<br>
Adjust learning rate by D_KL adaptively.<br>
Evaluate performance of current model against a Pure MCTS(simulation steps further when current model perform better).
