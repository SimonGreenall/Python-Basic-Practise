import unittest #python自带的unittest模块提供了代码测试工具，我们引入它来给系统做测试
from BANK import Bank #引入我写好的银行类

class BankTestCase(unittest.TestCase):
    def setUp(self):
        self.user=Bank() #创建一个银行对象供下面测试使用
    
    def test_register_user(self):
        """开户功能能正确存储Eddie Hall,男,不存钱这样的客户么"""
        register_user=self.user.register_user()
        self.assertEqual(register_user['姓名'],'Eddie Hall')
        self.assertEqual(register_user['性别'],'男')
        self.assertEqual(register_user['余额'],0)
    
    def test_login_user(self):
        """登录功能能正确返回Eddie Hall,男,不存钱这样的客户么"""
        login_user=self.user.login_user()
        self.assertEqual(login_user['姓名'],'Eddie Hall')
        self.assertEqual(login_user['性别'],'男')
        self.assertEqual(login_user['余额'],0)
    
unittest.main()
