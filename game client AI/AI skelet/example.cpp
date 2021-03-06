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

const char *infotext="name=\"destruCtioNN\", author=\"Mat�� Bako\", version=\"0.1\", country=\"Slovakia\", www=\"http://www.google.sk\"";

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

bool send_buffer(char* buffer, unsigned int size)
{
	int iResult = send(ConnectSocket, (char*)buffer, size, 0);
	if (iResult == SOCKET_ERROR)
	{
		pipeOut("ERROR Send failed: %d", WSAGetLastError());
		closesocket(ConnectSocket);
		WSACleanup();
		return false;
	}
	return true;
}

void send_preloaded_move(int x, int y, int player)
{
	unsigned char buf[9] = { 255,255,255,x,y,1,255,255,255 };

	if (send_buffer((char*)buf, 9))
		pipeOut("MESSAGE coords sent x-%d y-%d", x, y);
}

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
	int iResult = shutdown(ConnectSocket, SD_BOTH);

	if (iResult == SOCKET_ERROR)
		pipeOut("ERROR Shutdown failed: %d\n", WSAGetLastError());

	closesocket(ConnectSocket);
	WSACleanup();
}

/*
* Initialize brain
*/

bool read_word(FILE* file, char* buffer, int size)
{
	int c, index = 0;

	while (!iswspace(c = getc(file)) && c != EOF && index < size)
		buffer[index++] = c;
	printf("%d\r\n", index);
	return c == (int)EOF;
}

void load_addr(std::string& ip, std::string& port)
{
	char buffer[30] = { 0 };
	pipeOut("MESSAGE brain init");
	char* filename = ".\pbrain-destruCtioNN.cfg";
	FILE* file;
	pipeOut("MESSAGE calling fopen");

	if (fopen_s(&file, filename, "r"))
	{
		int errn = errno;
		char* err = strerror(errn);
		pipeOut("MESSAGE config file could not be opened: %s", err);
		pipeOut("ERROR config file could not be opened: %s", err);
		return;
	}

	pipeOut("MESSAGE file opened");
	ip = "", port = "";

	for (int i = 0; i < 2; i++)
	{
		bool end = read_word(file, buffer, 30);
		if (end)
			break;

		switch (i)
		{
		case 0:
			ip = std::string(buffer);
			break;
		case 1:
			port = std::string(buffer);
			break;
		}

		for (int j = 0; j < 30; j++)
			buffer[j] = 0;
	}
	fclose(file);
}

void brain_init() 
{
	pipeOut("MESSAGE brain init");
	if (width != 19 || height != 19)
	{
		pipeOut("ERROR size of the board");
		return;
	}
	seed = start_time;

	std::string IP = SERVER_IP, port = DEFAULT_PORT;

	//load_addr(IP, port);

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
}

void brain_restart()
{
	pipeOut("MESSAGE brain restart");

	int x,y;
	for(x=0; x<width; x++)
		for(y=0; y<height; y++)
			board[y * MAX_BOARD + x]=0;

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
	return x>=0 && y>=0 && x<width && y<height && board[y * MAX_BOARD + x]==0;
}

void brain_my(int x,int y, bool sendToClient)
{
	pipeOut("MESSAGE brain my");
	//if(isFree(x,y))
		board[y * MAX_BOARD + x]=1;

		if (sendToClient)
			send_preloaded_move(x, y, 0);

	//else
		//pipeOut("MESSAGE board %d", board[y * MAX_BOARD + x]);
		//pipeOut("ERROR my move [%d,%d]",x,y);
}

void brain_opponents(int x,int y, bool loadingBoard)
{
	pipeOut("MESSAGE brain oponents");
	if (isFree(x, y))
	{
		board[y * MAX_BOARD + x] = 2;

		if (loadingBoard)
			send_preloaded_move(x, y, 1);
		else
		{
			// encapsulate data
			unsigned char buf[6] = { 255,255,x,y,255,255 };

			if (send_buffer((char*)buf, 6))
				pipeOut("MESSAGE coords sent x-%d y-%d", x, y);
		}
	}
	else
		pipeOut("ERROR opponents's move [%d,%d]",x,y);
}

void brain_block(int x,int y)
{
	if(isFree(x,y))
		board[y * MAX_BOARD + x]=3;
	else
		pipeOut("ERROR winning move [%d,%d]",x,y);
}

int brain_takeback(int x,int y)
{
	if(x>=0 && y>=0 && x<width && y<height && board[y * MAX_BOARD + x]!=0)
	{
		board[y * MAX_BOARD + x]=0;
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

	unsigned char buf[6] = { 255,255,'m','o',255,255 };
	if (!send_buffer((char*)buf, 6))
		return;


	pipeOut("MESSAGE Waiting for response");
	// recieve data (x,y)
	int iResult = recv(ConnectSocket, (char*)coords, 2 * sizeof(char), 0);
	if (iResult > 0)
		pipeOut("MESSAGE Bytes received: %d\n", iResult);
	else if (iResult == 0)
		pipeOut("MESSAGE Connection closed\n");
	else
		pipeOut("ERROR recv failed: %d\n", WSAGetLastError());

	pipeOut("MESSAGE recieved %d %d\n", coords[::x], coords[::y]);
		
	// disconnect sequence initiated
/*	if (coords[::x] == 'k' && coords[::y] == 'o')
	{
		pipeOut("MESSAGE Recieved ko\n");
		unsigned char buf[6] = { 255,255,255,255,255,255 };

		int iResult = send(ConnectSocket, (char*)buf, sizeof(buf), 0);
		pipeOut("MESSAGE answer sent\n");
		if (iResult == SOCKET_ERROR)
		{
			pipeOut("ERROR Send failed: %d", WSAGetLastError());
			closesocket(ConnectSocket);
			WSACleanup();
			return;
		}

		socket_close();
		pipeOut("ERROR server closed connection", x, y);
		return;
	}*/

	pipeOut("MESSAGE x-%d y-%d\n",coords[0],coords[1]);
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
