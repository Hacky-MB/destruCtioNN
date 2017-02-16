#include "pisqpipe.h"


#include <windows.h>
#include <winsock2.h>
#include <iostream>
#include <ws2tcpip.h>
#include <stdlib.h>
#include <stdio.h>

// Need to link with Ws2_32.lib
#pragma comment (lib, "Ws2_32.lib")
// #pragma comment (lib, "Mswsock.lib")


#include <fstream>
#include <string>

const char *infotext="name=\"destruCtioNN\", author=\"Matúš Bako\", version=\"0.1\", country=\"Slovakia\", www=\"http://www.google.sk\"";

#define SERVER_IP "192.168.56.1"
#define DEFAULT_PORT "27015"

const char x = 0;
const char y = 1;
unsigned char coords[2];

#define MAX_BOARD 19
int board[361];
static unsigned seed;
SOCKET ConnectSocket;
WSADATA wsaData;
	
/*
 * Initialize socket
 */

int socket_init(std::string IP, std::string port)
{
	//create socket
	int iResult;

	//pipeOut("MESSAGE call WSAStartup()");

	// Initialize Winsock
	iResult = WSAStartup(MAKEWORD(2, 2), &wsaData);
	if (iResult != 0)
	{
		pipeOut("ERROR couldn't create socket, error - %d", iResult);
		return EXIT_FAILURE;
	}
	//pipeOut("MESSAGE done WSAStartup()");

	struct addrinfo *result = NULL,
		*ptr = NULL,
		hints;

	ZeroMemory(&hints, sizeof(hints));
	hints.ai_family = AF_INET;
	hints.ai_socktype = SOCK_STREAM;
	hints.ai_protocol = IPPROTO_TCP;

	//pipeOut("MESSAGE call getaddrinfo()");

	// translate host name to address
	iResult = getaddrinfo(IP.c_str(), port.c_str(), &hints, &result);
	if (iResult != 0)
	{
		pipeOut("ERROR error at getaddrinfo(), code - %d", iResult);
		WSACleanup();
		return EXIT_FAILURE;
	}

	//pipeOut("MESSAGE done getaddrinfo()");
	//pipeOut("MESSAGE call socket() and connect()");

	// Attempt to connect to an address until one succeeds
	for (ptr = result; ptr != NULL; ptr = ptr->ai_next) {

		// Create a SOCKET for connecting to server
		ConnectSocket = socket(ptr->ai_family, ptr->ai_socktype,
			ptr->ai_protocol);
		if (ConnectSocket == INVALID_SOCKET) {
			pipeOut("ERROR socket failed with error: %ld\n", WSAGetLastError());
			WSACleanup();
			return EXIT_FAILURE;
		}

		// Connect to server.
		iResult = connect(ConnectSocket, ptr->ai_addr, (int)ptr->ai_addrlen);
		if (iResult == SOCKET_ERROR) {
			closesocket(ConnectSocket);
			ConnectSocket = INVALID_SOCKET;
			continue;
		}
		break;
	}

	//pipeOut("MESSAGE done connect() %d", ConnectSocket == INVALID_SOCKET);

	// free output from getaddrinfo
	freeaddrinfo(result);

	if (ConnectSocket == INVALID_SOCKET)
	{
		WSACleanup();
		return EXIT_FAILURE;
	}

	//pipeOut("MESSAGE Socket created");
	return EXIT_SUCCESS;
}

void socket_close()
{
	int iResult = shutdown(ConnectSocket, SD_SEND);

	if (iResult == SOCKET_ERROR)
		pipeOut("ERROR Shutdown failed: %d\n", WSAGetLastError());

	closesocket(ConnectSocket);
	WSACleanup();
}

/*
* Initialize brain
*/

void brain_init() 
{

	pipeOut("MESSAGE brain init");
	if (width != 19 || height != 19)
	{
		pipeOut("ERROR size of the board");
		return;
	}
	seed=start_time;
	/*
	// read IP from file
	std::ifstream config_file("pbrain-destruCtioNN.cfg",'r');
	if (!config_file.is_open())
	{
		pipeOut("ERROR couldn't open config file");
		return;
	}*/

	std::string IP = SERVER_IP, port = DEFAULT_PORT;/*
	char c;
	char tmp[20] = { 0 };

	// read IP
	//
	config_file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
	try
	{
		config_file.get(c);
		//config_file.getline(tmp, 2);

		for (int i = 0; i < 12; i++)
		{

			IP += c;
			config_file.get(c);
		}

	}
	catch (std::ifstream::failure& e)
	{
		if (!config_file.eofbit)
		{
			pipeOut("ERROR couldn't read from file");
			return;
		}
	}*/

	//pipeOut("MESSAGE nacital som %c %s ",IP[0],IP);
	/*while (c != ' ' && c != '\r' && c != '\n')
	{
		pipeOut("MESSAGE nacital som %c",c);
		IP += c;
		config_file.get(c);
	}*/
	//pipeOut("MESSAGE ip - %s",IP);
	
	/*pipeOut("DEBUG IP - %s",IP);

	// skip ' '
	config_file.get(c);

	while (c != '\r' && c != '\n')
	{
		port += c;
		config_file.get(c);
	}
	config_file.close();
  
	pipeOut("DEBUG port - %s",port);*/

	if (!socket_init(SERVER_IP, DEFAULT_PORT))
		pipeOut("OK");


	char buffer[] = "in";
	int iResult = send(ConnectSocket, buffer, (int)strlen(buffer), 0);
	if (iResult == SOCKET_ERROR) 
	{
		pipeOut("ERROR send failed with error: %d\n", WSAGetLastError());
		closesocket(ConnectSocket);
		WSACleanup();
	}

	coords[::x] = 's';
	coords[::y] = 't';
}

void brain_restart()
{
	pipeOut("MESSAGE brain restart");

	int x,y;
	for(x=0; x<width; x++)
		for(y=0; y<height; y++)
			board[x * MAX_BOARD + y]=0;

	char buffer[] = "in";
	int iResult = send(ConnectSocket, buffer, (int)strlen(buffer), 0);
	if (iResult == SOCKET_ERROR)
	{
		pipeOut("ERROR send failed with error: %d\n", WSAGetLastError());
		closesocket(ConnectSocket);
		WSACleanup();
	}

	pipeOut("OK");
}

int isFree(int x, int y)
{
	return x>=0 && y>=0 && x<width && y<height && board[x * MAX_BOARD + y]==0;
}

void brain_my(int x,int y)
{

	pipeOut("MESSAGE brain my");
	if(isFree(x,y))
		board[x * MAX_BOARD + y]=1;
	else
		pipeOut("ERROR my move [%d,%d]",x,y);
}

void brain_opponents(int x,int y) 
{
	pipeOut("MESSAGE brain oponents");
	if (isFree(x, y))
	{
		coords[::x] = x;
		coords[::y] = y;
		board[x * MAX_BOARD + y] = 2;
	}
	else
		pipeOut("ERROR opponents's move [%d,%d]",x,y);
}

void brain_block(int x,int y)
{
	if(isFree(x,y))
		board[x * MAX_BOARD + y]=3;
	else
		pipeOut("ERROR winning move [%d,%d]",x,y);
}

int brain_takeback(int x,int y)
{
	if(x>=0 && y>=0 && x<width && y<height && board[x * MAX_BOARD + y]!=0)
	{
		board[x * MAX_BOARD + y]=0;
		return 0;
	}
	return 2;
}

unsigned rnd(unsigned n)
{
	seed=seed*367413989+174680251;
	return (unsigned)(UInt32x32To64(n,seed)>>32);
}

void brain_turn()
{

	pipeOut("MESSAGE brain turn");
	/*int x,y,i;
	i=-1;
	do{
		x=rnd(width);
		y=rnd(height);
		i++;
		if(terminateAI)
			return;
	}
	while(!isFree(x,y));

	if(i>1)
		pipeOut("DEBUG %d coordinates didn't hit an empty field",i);
	do_mymove(x,y);*/

	// posles pole sizeof(int)*19*19, s nacitanym polom
	// citas sizeof(int)*2

	//send data (last enemy move)

	// encapsulate data
	unsigned char buf[6] = { 255,255,coords[0],coords[1],255,255 };

	int iResult = send(ConnectSocket, (char*)buf,sizeof(buf), 0);
	if (iResult == SOCKET_ERROR)
	{
		pipeOut("ERROR Send failed: %d", WSAGetLastError());
		closesocket(ConnectSocket);
		WSACleanup();
		return;
	}
	pipeOut("MESSAGE coords sent %d %d", coords[::x], coords[::y]);

	// recieve data (x,y)
	iResult = recv(ConnectSocket, (char*)coords, 2 * sizeof(char), 0);
	if (iResult > 0)
		pipeOut("MESSAGE Bytes received: %d\n", iResult);
	else if (iResult == 0)
		pipeOut("MESSAGE Connection closed\n");
	else
		pipeOut("ERROR recv failed: %d\n", WSAGetLastError());

		pipeOut("MESSAGE recieved %d %d\n", coords[0], coords[1]);
		
	socket_close();
	socket_init(SERVER_IP, DEFAULT_PORT);

	do_mymove(coords[::x], coords[::y]);
}

void brain_end()
{
	socket_close();
}

#ifdef DEBUG_EVAL
#include <windows.h>

void brain_eval(int x,int y)
{
	HDC dc;
	HWND wnd;
	RECT rc;
	char c;
	wnd=GetForegroundWindow();
	dc= GetDC(wnd);
	GetClientRect(wnd,&rc);
	c=(char)(board[x][y]+'0');
	TextOut(dc, rc.right-15, 3, &c, 1);
	ReleaseDC(wnd,dc);
}

#endif
