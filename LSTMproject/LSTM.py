import tensorflow as tf
import numpy as np
import pickle
import random
import matplotlib.pyplot as plt
data_file=open('youtube_train_data.pkl','rb')
t_data,t_labels=pickle.load(data_file)
data_file.close()
train_data=t_data[0:6999]
train_labels=t_labels[0:6999]
validation_data=t_data[7000:]
validation_labels=t_labels[7000:]
#print(np.array(validation_data).shape)
def initweight(shape):
	weight=tf.truncated_normal(shape,stddev=0.01)
	return weight
def initbias(shape):
	bias=tf.zeros(shape)
	return bias
def conv2d(x,w):
	return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding="VALID")
def maxpooling(x):
	return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID")
def dist(dat,newdat):
	dist=np.sqrt((dat[0]-newdat[0])**2+(dat[1]-newdat[1])**2)
	return dist
x=tf.placeholder('float32',[None,64,64,3])
y=tf.placeholder('float32',[None,10,14])
rnn_input=tf.placeholder('float32',[None,10,3872])
a=np.zeros([5,10,3872])
valid=np.zeros([1000,10,3872])
trai=np.zeros([1000,10,3872])
accuracy=np.zeros([7])
w1=tf.Variable(initweight([5,5,3,16]))
w2=tf.Variable(initweight([5,5,16,16]))
w3=tf.Variable(initweight([3,3,16,32]))
b1=tf.Variable(initbias([16]))
b2=tf.Variable(initbias([16]))
b3=tf.Variable(initbias([32]))
distance=np.zeros([1000,10,7,2])
num_units=100
batch_size=5
w_fc=tf.Variable(initweight([num_units,14]))
b_fc=tf.Variable(initbias([14]))
final_output=tf.zeros(shape=[batch_size,0,14])
final_output1=tf.zeros(shape=[1000,0,14])
clayer1=tf.nn.relu(conv2d(x,w1)+b1)
pool1=maxpooling(clayer1)
clayer2=tf.nn.relu(conv2d(pool1,w2)+b2)
pool2=maxpooling(clayer2)
clayer3=tf.nn.relu(conv2d(pool2,w3)+b3)
fullylayer=tf.reshape(clayer3,[-1,11*11*32])
lstm_cell=tf.nn.rnn_cell.LSTMCell(num_units)
h_val,_=tf.nn.dynamic_rnn(lstm_cell,rnn_input,dtype=tf.float32)
for i in np.arange(10):
	temp=tf.reshape(h_val[:,i,:],[batch_size,num_units])
	output=tf.matmul(temp,w_fc)+b_fc
	output=tf.reshape(output,[-1,1,14])
	final_output=tf.concat([final_output,output],axis=1)
loss=tf.losses.mean_squared_error(y,final_output)
optimizer=tf.train.AdamOptimizer(0.001).minimize(loss)
init=tf.global_variables_initializer()
for o in np.arange(10):
	temp=tf.reshape(h_val[:,i,:],[1000,num_units])
	output=tf.matmul(temp,w_fc)+b_fc
	output=tf.reshape(output,[-1,1,14])
	final_output1=tf.concat([final_output1,output],axis=1)
with tf.Session() as sess:
	sess.run(init)
	for q in range(1001):
		num=random.randint(0, len(train_data) - batch_size)
		batch_x=train_data[num:num+batch_size]
		batch_y=np.reshape(train_labels[num:num+batch_size],[batch_size,10,14])
		for i in range(batch_size):
			new_data=batch_x[i]
			a[i]=sess.run(fullylayer,feed_dict={x:new_data})
		nodes=sess.run(optimizer,feed_dict={rnn_input:a, y:batch_y})
		if q%1000==0:
		#	loo=sess.run(loss,feed_dict={rnn_input:a, y:batch_y})
		#	print(loo)
			for m in range(1000):
				valid[m]=sess.run(fullylayer,feed_dict={x:validation_data[m]})
				trai[m]=sess.run(fullylayer,feed_dict={x:train_data[m]})
			k=sess.run(final_output1,feed_dict={rnn_input:valid})
			#c=sess.run(final_output1,feed_dict={rnn_input:trai})
			#train_result=np.reshape(c,[1000,10,7,2])
			#cost=np.mean((train_result-train_labels[0:1000])**2)
			
	#print(k[1][1])
			result=np.reshape(k,[1000,10,7,2])
			testloss=np.mean((result-validation_labels[0:1000])**2)
			#print(cost,testloss)
	#print(result[2][3][:,0],result[2][3][:,1])
	#for j in range(1000):
	#	for l in range(10):
	#		distance[j,l,0]=dist(result[j][l][0],validation_labels[j][l][0])
	#		distance[j,l,1]=dist(result[j][l][1],validation_labels[j][l][1])
	#		distance[j,l,2]=dist(result[j][l][2],validation_labels[j][l][2])
	#		distance[j,l,3]=dist(result[j][l][3],validation_labels[j][l][3])
	#		distance[j,l,4]=dist(result[j][l][4],validation_labels[j][l][4])
	#		distance[j,l,5]=dist(result[j][l][5],validation_labels[j][l][5])
	#		distance[j,l,6]=dist(result[j][l][6],validation_labels[j][l][6])
	#for t in np.arange(0,20.5,0.5):
	#	print(t)
	#	for u in range(7):
	#		count=distance[:,:,u]<t
	#		count=np.float32(count)
	#		accuracy[u]=np.mean(count)
	#	print(accuracy)
	
	#result=np.reshape(final_output[1],[5,10,7,2])
	#result=np.reshape(k[1][5],[7,2])
	#print(result)
	#print(train_labels[0][1])
	#plt.imshow(np.uint8(validation_data[40][5]))
	#plt.plot(result[40][5][:,0],result[40][5][:,1],'ro')
	#plt.plot(train_labels[0][1][:,0],train_labels[0][1][:,1],'ro')
	#plt.show()
	plt.figure(figsize=(8,8))
	plt.subplot(2,2,1)
	plt.imshow(np.uint8(validation_data[4][6]))
	plt.plot(result[4][6][:,0],result[4][6][:,1],'ro')
	plt.subplot(2,2,2)
	plt.imshow(np.uint8(validation_data[150][2]))
	plt.plot(result[150][2][:,0],result[150][2][:,1],'ro')
	plt.subplot(2,2,3)
	plt.imshow(np.uint8(validation_data[100][1]))
	plt.plot(result[100][1][:,0],result[100][1][:,1],'ro')
	plt.subplot(2,2,4)
	plt.imshow(np.uint8(validation_data[130][2]))
	plt.plot(result[130][2][:,0],result[130][2][:,1],'ro')			
	plt.show()