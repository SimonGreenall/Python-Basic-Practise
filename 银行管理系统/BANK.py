import json

class Bank():#设计一个银行管理系统
    def __init__(self,bank_name='猪猪存钱罐'):#初始化，默认名字猪猪存钱罐
        self.bank_name=bank_name #把形参赋给self.bank_name
        self.users=[] #创建一个列表用来存放客户信息（姓名，性别，存款），客户信息用字典方式存储
        self.file_name='users.json' #设置文件系统作为数据库用来保存客户信息，之后每次存款取款都需要和数据库交互，users.json在首位客户开户后自动创建
        print("欢迎来到"+self.bank_name) #欢迎语
        while True:
              f1=input("您好，是否已经开户？(是/否/q.退出银行系统)")
              if(f1=='是'):
                    while True:
                          f2=input("请问要办理什么业务(1.存款 2.取款 q.退出业务系统)")
                          if(f2=='1'):
                                self.deposit(self.login_user())
                                f3=input("是否要办理其它业务？(是/否)")
                                if(f3=='是'):
                                      continue
                                else:
                                      break
                          elif(f2=='2'):
                                self.withdraw(self.login_user())
                                f3=input("是否要办理其它业务？(是/否)")
                                if(f3=='是'):
                                      continue
                                else:
                                      break
                          elif(f2=='q'):
                                break
                          else:
                                print("输入非法!请重新输入")
                                continue
              elif(f1=='否'):
                    print("那么请先开户")
                    NEW_USER=self.register_user()
                    
              elif(f1=='q'):
                    break
              else:
                    print("请不要输入非法字符!")
                    continue

    def register_user(self): #开户
        self.register_user_name=input("姓名:") #输入姓名并保存到变量
        self.register_user_sex=input("性别:") #输入性别并保存到变量
        self.register_user_money=0 #默认余额为0
        while True: #设置循环的目的在于假如用户不输入（是/否），输入别的也可以应对，提升鲁棒性
                    flag=input("是否要存钱?(是/否)") #设置标志
                    if flag=='是':
                         self.register_user_money+=int(input("存钱数额:")) #存钱,输入的数额是字符串，要转化为整型
                         new_user={'姓名':self.register_user_name,'性别':self.register_user_sex,'余额':self.register_user_money} #创建字典保存用户信息
                         self.users.append(new_user) #将做好的字典保存到用来存放客户信息的列表中
                         with open(self.file_name,'w') as file_object: #以写入模式打开文件
                               for user in self.users: #遍历用来存储客户信息的列表，把得到的每个客户的字典信息存储到user中
                                     file_object.write(json.dumps(user)) #将客户的字典信息用json.dumps()转换为字符串存入users.json文件中
                         return new_user #把新注册好的用户返回给函数，开户结束
                    elif flag=='否':
                         new_user={'姓名':self.register_user_name,'性别':self.register_user_sex,'余额':self.register_user_money}
                         self.users.append(new_user) #不存钱就直接把客户信息存到列表里
                         with open(self.file_name,'w') as file_object: 
                               for user in self.users:
                                     file_object.write(json.dumps(user)) #将客户存入users.json文件中,注意这里是不能直接将字典写到文件里的，要用json库函数对字典进行转换，转换成字符串
                         return new_user #把新注册好的用户返回给函数，开户结束
                    else:
                         print("你输入的不是规定字符！请重新输入！")
                         continue #输入非法字符就返回循环开头重新来过
    
    def login_user(self): #登录
          self.login_user_name=input("姓名：")
          try:
               with open(self.file_name) as file_object:
                     lines=file_object.readlines()   #检索保存的已注册的用户信息，把已注册用户的字典信息读到lines列表里，readlines()返回的是一个字符串列表
                     for line in lines: #遍历lines，找找看有没有刚才输入的客户对应的信息
                          user=json.loads(line) #lines是一个字符串列表，line则是列表中的每一个字符串，也是客户信息，用json.loads()把它转换成字典
                          if(self.login_user_name==user['姓名']): #如果找到了就提示登录成功，并且把找到的客户字典信息返回给函数
                            print("登录成功！")
                            return user #把客户字典信息返回给函数，函数功能到此结束
                     while True: #如果没有从数据库中找到客户信息，说明客户还没开户，就进入到这一步
                           flag=input("未开户,是否需要开户?(是/否)") #设置标志位，判断是否要开户
                           if flag=='是': 
                                 return self.register_user() #要开户就调用开户函数去开户，开好户之后把新客户返回给登录
                           elif flag=='否':
                                 break #不开户，登录函数也直接结束
                           else:
                                print("你输入的不是规定字符！请重新输入！")
                                continue #返回循环开头重新来过
          except FileNotFoundError: #前面已经用try关键字去捕获异常，因为这里有可能出现还没开户录信息，数据库还没建立的情况，这里是异常处理
                print("你还没开户！请先开户！")
                return self.register_user()

    def deposit(self,user):  #存钱
          self.add=int(input("存多少:")) #输入存多少钱，将存的钱放入self.add,注意输入的是字符串，要转换为整型
          with open(self.file_name,'w+') as file_object: #以读写模式打开数据库文件，并把它作为file_object对象
                lines=file_object.read() #从文件中读取每一行，并将它保存到lines列表中，readlines()返回一个字符串列表
                for line in lines: #遍历这个列表，看看要存钱的客户有没有开户
                      USER=json.loads(line) #将存储的字符串转换为字典
                      if(user['姓名']==USER['姓名']): #如果找到了就说明开过户了
                            USER['余额']=str(int(USER['余额'])+self.add) #就直接找到对应的余额然后把要存的钱加上去,因为余额是字符串保存的，所以要先转换为整型和self.add相加，加完再转换为字符串存回去
                            json.dumps(USER,file_object)
                            print("存款成功!") #提示已经存款成功

    def withdraw(self,user): #取钱
          self.reduce=int(input("取多少:")) #输入取多少钱，将要取的钱放入self.reduce,注意输入的是字符串，要转换为整型
          with open(self.file_name,'w+') as file_object: #以读写模式打开数据库文件，并把它作为file_object对象
                lines=file_object.read() #从文件中读取每一行，并将它保存到lines列表中
                for line in lines: #遍历这个列表，看看要取钱的客户有没有开户
                      USER=json.loads(line) #将存储的字符串转换为字典
                      if(user['姓名']==USER['姓名']): #如果找到了就说明开过户了
                            if(self.reduce<=int(USER['余额'])): #这里再多加一个判定，就是取钱额不能超过余额
                                  USER['余额']=str(int(USER['余额'])-self.reduce)
                                  json.dumps(USER,file_object)
                                  print("取款成功！")
                            elif(self.reduce>int(USER['余额'])):
                                  print("你没这么多钱！取款失败")