function [W , final_train_lda, aim_v]= fun_CDL_Train(data, d1, d2, label)
  % author：WR
  % date：2017.03.02
  % function：LDA
  %% ldA之前先进行pca操作
  a=size(data,2);%确定个数
%   [c,d]=size(data{1,1});%取出每一个图像集的尺寸
%   train_matrix=zeros(c*d,a);%把cell转换成矩阵
%   for i=1:a
%     temp=data{i};
%     temp=reshape(temp,c*d,1);%拉成一个列向量
%     train_matrix(:,i)=temp;%cell数组就变成了矩阵的形式
%   end
  mean_pca=mean(data,2);%训练样本的均值
  mean_pca=repmat(mean_pca,1,a);%拉成矩阵
  center_train=data-mean_pca;%样本中心化操作
  Cov_train=(center_train*center_train')/(a-1);%训练样本的协方差矩阵
  [V,U]=eig(Cov_train);%特征值分解
  [dummy,index]=sort(diag(-U));
  v_sort=V(:,index);%按照特征值的大小把特征向量重新排序
  aim_v=v_sort(:,1:d1);%取出所需的前的的d1个特征向量
  new_data=aim_v'*data;%pca降维后的数据
%    [coeff,~,latent]=pca(data');
%    aim_v = coeff(:,1:d1);
%    new_data = aim_v'*data;
  %% lda的过程
  % step1: 计算类内离散度矩阵
  nclasses=unique(label); %统计类别数 
  num=length(nclasses);% 8
  data_lda=cell(1,num);
  for j=1:num
      data_lda{j}=new_data(:,(j-1)*5+1:j*5);%取出每一类放到cell中，5表示是每一类的训练样本数，此处简写，未用变量
  end
  for l=1:num
     mean_class(:,l)=mean(data_lda{l},2);
  end
  sw=zeros(size(mean_class,1),size(mean_class,1));%每一类的类内离散度矩阵
  Sw=zeros(size(mean_class,1),size(mean_class,1));%总的类内离散度矩阵
  for m=1:num
     for n=1:5
         temp=data_lda{m};%取出每一类
         sw=sw+(temp(:,n)-mean_class(:,m))*(temp(:,n)-mean_class(:,m))';
     end
     Sw=Sw+sw;
  end
 % step2: 计算类间离散度矩阵
  Sb=zeros(size(mean_class,1),size(mean_class,1));%总的类间离散度矩阵
  mean_all=mean(new_data,2);%训练样本总的均值
  for p=1:num
      Sb=Sb+5*(mean_class(:,p)-mean_all)*(mean_class(:,p)-mean_all)';
  end
 %step3 : 计算投影矩阵
  [V1,U1]=eig(Sb,Sw);
  [dummy1,index1]=sort(diag(-U1));
  v_sort1=V1(:,index1);%按照特征值的大小把特征向量重新排序
  W=v_sort1(:,1:d2);%取出所需的前的的d2个特征向量
  final_train_lda=W'*new_data;%最终的训练数据
end 