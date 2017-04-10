#include "Log.h"

#include <iostream>
#include <Windows.h>

void platformLogMessage(int level, const char* message) {
	HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);

	switch (level)
	{
	case 3:
		SetConsoleTextAttribute(hConsole, BACKGROUND_RED | BACKGROUND_INTENSITY | FOREGROUND_RED | FOREGROUND_BLUE | FOREGROUND_GREEN | FOREGROUND_INTENSITY);
		break;
	case 2:
		SetConsoleTextAttribute(hConsole, FOREGROUND_RED | FOREGROUND_INTENSITY);
		break;
	case 1:
		SetConsoleTextAttribute(hConsole, FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_INTENSITY);
		break;
	}

	printf("%s", message);
	SetConsoleTextAttribute(hConsole, FOREGROUND_RED | FOREGROUND_BLUE | FOREGROUND_GREEN);
}

void waitHere() {
	std::cin.get();
}