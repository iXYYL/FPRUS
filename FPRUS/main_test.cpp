/*
 * Author: Corfox
 * Date: 2015.10.22
 */

#include <iostream>
#include <cstdlib>
#include "eigensketch_test.h"
#include "dataset_test.h"
#include "eigensketchsst_test.h"

using std::cout;
using std::endl;
using std::cin;
using std::exit;

int main(void)
{
	cout << "���������´���ѡ��������ࡣ" << endl;
	cout << "e or E - Exit" << endl;
	cout << "1 - Eigensketch����" << endl;
	cout << "2 - DataSet����" << endl;
	cout << "3 - EigensketchSST����" << endl;
	char key;
	cin >> key;
	switch (key)
	{
	case 'E':
	case 'e':
		exit(0);
		break;
	case '1':				//Eigensketch����
		cout << " 0 - exit; 1 - �����任; 2 - ʶ��; 3 - siftʶ��" << endl;
		cin >> key;
		if (key == '1')
			eigensketch_test();
		else if (key == '2')
		for (int i = 1; i < 11; ++i)
			eigensketch_recognition_test(i);
		else if (key == '3')
		for (int i = 1; i < 11; ++i)
			eigensketch_sift_match_test(i);
		else
			exit(0);
		break;
	case '2':				//DataSet����
		dataset_test();
		break;
	case '3':				//EigensketchSST����
		eigensketchsst_test();
		break;
	default:
		break;
	}
}