#include	<stdio.h>
#include <iostream>
#include <string>
#include <atlstr.h>
#include <locale>
#include <cstdlib>
#include "HearA.h"
#include "cu.cuh"

int main()
{
	//std::string strVar1 = {"1,2,3"};
	//int iRet = 0, index = 0 , num=0;

	//while (true) {
	//	iRet = strVar1.find(',', index);
	//	if (iRet != -1) {
	//		index = iRet + 1;
	//		num++;
	//	}
	//	else {
	//		break;
	//	}
	//}num++;
	//printf_s("Hello World!\n");	//	输出hello World!
	//std::cout << "num: " << num << "  index: " << index << std::endl;\

	//	CString s1 = _T("This ");        // Cascading concatenation
	//s1 += _T("is a ");
	//CString s2 = _T("test");
	//CString s3 = ( "cool ");
	//CString message = s1 + _T("big ") + s3 + s2;
	//std::wcout << (const wchar_t*)message << std::endl;
	// Message contains "This is a big test".

	//CString cs("meow");
	//std::wcout << (const wchar_t*)cs <<std:: endl;	

		//	暂停程序运行，查看输出结果
					
	
	getResultFormcGpuCal();
	//basicDem::cGpuCal ob{};
	//ob.sumMatrix();//////这里竟然忘了！！！！！！！！！！！！！！！！！！！！1
	//double* temp = ob.c;
	//for (int i{ 0 }; i < 3; i++) {
	//	std::cout << temp[i] << std::endl;
	//}
	//std::cout << temp[0] << std::endl;
	system("pause");
	
//	getchar();		
	return 0;
}

//
//#include <iostream>
//#include <vector>
//#include "cuda_test.cuh"
//
//void main()
//{
//	GpuDeviceInfo();
//	//test_bgr2gray(0);
//	system("pause");
//
//}
