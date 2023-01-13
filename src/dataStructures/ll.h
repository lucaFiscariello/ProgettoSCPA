#ifndef LL_H_INCLUDED
#define LL_H_INCLUDED

#include <stdio.h>
#include <string.h>

// The fundamental element of linked list
typedef struct structNode{

	void *value;                // Value of node.
	struct structNode *next;    // Pointer to next node of list

} Node;

// Creates a node with given value
Node *newNode (void *value);

// prints content of linked list using given format string
#define printLL(head, format, type) \
	do { \
		Node *current = head; \
		char *buffer = calloc(strlen(format) + 3, sizeof(char)); \
		while (current != NULL){ \
			strcpy(buffer, format); \
			strcat(buffer, ", "); \
			printf(buffer, *(type*)(current -> value)); \
			current = current -> next; \
		} \
		printf("\n"); \
		free(buffer); \
	} while (0); \


//void printLL (Node *current, char *format);

// frees memory associated with ll
void destroyLL (Node *current);

/** creates a node with given value and adds it to the end of list. 
 * @param currentAddr address of pointer to head of list. If the address points to null,
 * the new node will be the head of the list .
*/
void appendLL (Node **currentAddr, void *value);

// Stores value of node at given index in destination and removes such node from list. If destination is NULL the value is not stored
int popLL(Node **headAddr, int index, void **destination);

// Returns leńgth of list
int lenLL(Node *current);

// stores in *node a pointer to node in head corresponding to given index
int getLL(Node *head, int index, Node **node);

#endif // LL_H_INCLUDED
