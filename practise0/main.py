while True:
    try:
        a = float(input("请输入第一个数字(输入q退出系统)"))
        b = float(input("请输入第二个数字(输入q退出系统)"))
        if(a=='q'):
            break
        else:
            print(a/b)
    except ZeroDivisionError:
        print("分母不可为0,重新输入！")
        continue
    except ValueError:
        print("请不要输入非法字符！重新输入！")
        continue
