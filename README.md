# Page-Rank-Algorithms

# HITS Algorithm
It is a method of link analysis, which is an acronym for HITS
<b>Hyperlink-Induced Topic Search</b>.

>### HOW TO RUN?

<b>Input:</b><br>
Enter the Query:<br>
The Query word is entered here.<br>

<b>Output:</b><br>
It gives us the HubScores and AuthorityScores of top Hubs and Authorities for the given query given in the input.<br>

<b>The following are printed :</b> 
*  Hubscores for given query
*  Order of the nodes according to hub scores
*  Authority Scores for given query
*  Order of the nodes according to Authority scores


# Page Ranking Algorithm
It is a link analysis which assigns to every node in
the web graph a numerical score between 0 and 1.
The PageRank of a node will depend on the link structure of the web graph.

>### HOW TO RUN?
<b>Input.txt:</b>

First Line includes the Number of nodes.<br>
Second Line includes the Number of Edges in the graph.<br>
From third line onwards we give the edges in the form of(source,destination).<br>

<b>The following are printed :</b> 
*  Transition Matrix without Teleportation.
*  Page Rank vector using Linear Algebra Method.
*  Page Rank vector using Iterative Method.
*  Transition Matrix with Teleportation.
*  Page Rank vector using Linear Algebra Method.
*  Page Rank vector using Iterative Method.


<b>Note: </b>
* We considered the graph as 1-based index means node number starts from 1 not 0.<br>
* Input file name is <b>input.txt</b> and file is present in the <b>same folder</b>.


