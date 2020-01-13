The figure froom https://blog.csdn.net/artprog/article/details/61614452
& https://zh-tw.coursera.org/learn/ntumlone-mathematicalfoundations （NTU Prof.HT Lin）     
http://terrence.logdown.com/posts/290508-python-simple-perceptron-learning-algorithm-implementations
<div align=center> <img src="https://github.com/AvisChiu/Machine_learning_practice/blob/master/PLA/pla.png" width="800",height="600"/></div>
<br/>

<div align=center> <img src="https://github.com/AvisChiu/Machine_learning_practice/blob/master/PLA/pla2.PNG" width="800",height="600"/></div>
<br/>


Some notes of PLA
---
* 資料的標簽均爲 -1， 1
* numpy sign： > 0 --> 1 ; < 0 --> 0; ==0 --> 0
* dot: 兩個向量做内積，再作 np.sign 運算， 目的就是看是否夾角過大（夾角過大表示不是同一類）
* 更新方法： w <-- w + yx (如上圖， y 是 1 或 -1， 所以沒所謂， 本來應該是 w+x)
* 内積是一個確定的值，作 np.sign 是爲了落在 -1 和 1，
* 最後得到一個法向量 w ，根據法向量找出直綫 Ax+Bx+C=0 （PLA只能是一條直綫，因爲是綫性分類器）
* 假設法向量為 (m,n), 那麽直綫的斜率為 （-m/n）, y 軸的截距為 (-C/n), C 已經知道是 1 了。
