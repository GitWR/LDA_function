function [W , final_train_lda, aim_v]= fun_CDL_Train(data, d1, d2, label)
  % author��WR
  % date��2017.03.02
  % function��LDA
  %% ldA֮ǰ�Ƚ���pca����
  a=size(data,2);%ȷ������
%   [c,d]=size(data{1,1});%ȡ��ÿһ��ͼ�񼯵ĳߴ�
%   train_matrix=zeros(c*d,a);%��cellת���ɾ���
%   for i=1:a
%     temp=data{i};
%     temp=reshape(temp,c*d,1);%����һ��������
%     train_matrix(:,i)=temp;%cell����ͱ���˾������ʽ
%   end
  mean_pca=mean(data,2);%ѵ�������ľ�ֵ
  mean_pca=repmat(mean_pca,1,a);%���ɾ���
  center_train=data-mean_pca;%�������Ļ�����
  Cov_train=(center_train*center_train')/(a-1);%ѵ��������Э�������
  [V,U]=eig(Cov_train);%����ֵ�ֽ�
  [dummy,index]=sort(diag(-U));
  v_sort=V(:,index);%��������ֵ�Ĵ�С������������������
  aim_v=v_sort(:,1:d1);%ȡ�������ǰ�ĵ�d1����������
  new_data=aim_v'*data;%pca��ά�������
%    [coeff,~,latent]=pca(data');
%    aim_v = coeff(:,1:d1);
%    new_data = aim_v'*data;
  %% lda�Ĺ���
  % step1: ����������ɢ�Ⱦ���
  nclasses=unique(label); %ͳ������� 
  num=length(nclasses);% 8
  data_lda=cell(1,num);
  for j=1:num
      data_lda{j}=new_data(:,(j-1)*5+1:j*5);%ȡ��ÿһ��ŵ�cell�У�5��ʾ��ÿһ���ѵ�����������˴���д��δ�ñ���
  end
  for l=1:num
     mean_class(:,l)=mean(data_lda{l},2);
  end
  sw=zeros(size(mean_class,1),size(mean_class,1));%ÿһ���������ɢ�Ⱦ���
  Sw=zeros(size(mean_class,1),size(mean_class,1));%�ܵ�������ɢ�Ⱦ���
  for m=1:num
     for n=1:5
         temp=data_lda{m};%ȡ��ÿһ��
         sw=sw+(temp(:,n)-mean_class(:,m))*(temp(:,n)-mean_class(:,m))';
     end
     Sw=Sw+sw;
  end
 % step2: ���������ɢ�Ⱦ���
  Sb=zeros(size(mean_class,1),size(mean_class,1));%�ܵ������ɢ�Ⱦ���
  mean_all=mean(new_data,2);%ѵ�������ܵľ�ֵ
  for p=1:num
      Sb=Sb+5*(mean_class(:,p)-mean_all)*(mean_class(:,p)-mean_all)';
  end
 %step3 : ����ͶӰ����
  [V1,U1]=eig(Sb,Sw);
  [dummy1,index1]=sort(diag(-U1));
  v_sort1=V1(:,index1);%��������ֵ�Ĵ�С������������������
  W=v_sort1(:,1:d2);%ȡ�������ǰ�ĵ�d2����������
  final_train_lda=W'*new_data;%���յ�ѵ������
end 