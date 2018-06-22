//	printf - format and print data
//	Copyright (C) 90, 91, 92, 93, 1994 Free Software Foundation, Inc.
//
//	This program is free software; you can redistribute it and/or modify
//	it under the terms of the GNU General Public License as published by
//	the Free Software Foundation; either version 2, or (at your option)
//	any later version.
//
//	This program is distributed in the hope that it will be useful,
//	but WITHOUT ANY WARRANTY; without even the implied warranty of
//	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.	See the
//	GNU General Public License for more details.
//
//	You should have received a copy of the GNU General Public License
//	along with this program; if not, write to the Free Software
//	Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
//
// Orignal version by David MacKenzie <djm@gnu.ai.mit.edu>
// Modified to build and run on Windows platforms by ahodgkinson@novell.com

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <ctype.h>

#define isodigit(c) \
	((c) >= '0' && (c) <= '7')
	
#define hextobin(c) \
	((c) >= 'a' && (c) <= 'f' ? (c) - 'a' + 10 \
				: (c) >= 'A' && (c) <= 'F' \
				? (c) - 'A' + 10 : (c) - '0')
	
#define octtobin(c) \
	((c) - '0')

char *xmalloc();
void error();

static int 		exit_status = 0;
char *			program_name;

void usage(
	int			status);
	
int print_formatted(
	char *		format,
	int			argc,
	char **		argv);
	
int print_esc(
	char *		escstart);

void print_esc_char(
	char			c);
	
void print_esc_string(
	char *		str);

void print_direc(
	char *		start,
	int			length,
	int			field_width,
	int			precision,
	char *		argument);
	
unsigned long xstrtoul(
	char *		s);

long xstrtol( 
	char *		s);

double xstrtod(
	char *		s);

void verify( 
	char *		s,
	char *		end);

void error( 
	int			status,
	int			errnum,
	char *		message, ...);

/****************************************************************************
Desc:
****************************************************************************/
inline bool isxdigit(
	char		c)
{
	if( (c >= '0' && c <= '9') || 
		 (c >= 'a' && c <= 'f') ||
		 (c >= 'A' && c <= 'F'))
	{
		return( true);
	}
	
	return( false);
}

/****************************************************************************
Desc:
****************************************************************************/
void main(
	int			argc,
	char **		argv)
{
	char *		format;
	int 			args_used;
	
	program_name = argv[ 0];
	
	if( argc == 1)
	{
		fprintf (stderr, "Usage: %s format [argument...]\n", program_name);
		exit( 1);
	}
	
	format = argv[ 1];
	argc -= 2;
	argv += 2;
	
	do
	{
		args_used = print_formatted (format, argc, argv);
		argc -= args_used;
		argv += args_used;
	}
	while( args_used > 0 && argc > 0);
	
	exit (exit_status);
}

/****************************************************************************
Desc:
****************************************************************************/
void usage(
	int		status)
{
	if( status != 0)
	{
		fprintf (stderr, "Try `%s --help' for more information.\n", program_name);
	}
	else
	{
		printf( "Usage: %s FORMAT [ARGUMENT]...\n or: %s OPTION\n",
			program_name, program_name);
			
		printf( "\n"
				  "FORMAT controls the output as in C printf.  Interpreted sequences are:\n"
				  "\n"
				  "   \\\"          double quote\n"
				  "   \\0NNN        character with octal value NNN (0 to 3 digits)\n"
				  "   \\\\          backslash\n"
				  "   \\a           alert (BEL)\n"
				  "   \\b           backspace\n"
				  "   \\c           produce no further output\n"
				  "   \\f           form feed\n"
				  "   \\n           new line\n"
				  "   \\r           carriage return\n"
				  "   \\t           horizontal tab\n"
				  "   \\v           vertical tab\n"
				  "   \\xNNN        character with hexadecimal value NNN (1 to 3 digits)\n"
				  "\n"
				  "   %%%%          a single %%\n"
				  "   %%b           ARGUMENT as a string with `\\' escapes interpreted\n"
				  "\n"
				  "   and all C format specifications ending with one of\n"
				  "   diouxXfeEgGcs, with ARGUMENTs converted to proper type\n"
				  "   first.	 Variable widths are handled.\n");
	}
	 
	exit (status);
}

/****************************************************************************
Desc:	Print the text in FORMAT, using ARGV (with ARGC elements) for
		arguments to any `%' directives.
		
		Return the number of elements of ARGV used.
****************************************************************************/
int print_formatted(
	char *	format,
	int		argc,
	char **	argv)
{
	int 		save_argc = argc;		// Preserve original value
	char *	f;							// Pointer into `format'
	char *	direc_start;			// Start of % directive
	int 		direc_length;			// Length of % directive
	int 		field_width;			// Arg to first '*', or -1 if none
	int 		precision;				// Arg to second '*', or -1 if none

	for( f = format; *f; ++f)
	{
		switch( *f)
		{
			case '%':
			{
				direc_start = f++;
				direc_length = 1;
				field_width = precision = -1;
				
				if( *f == '%')
				{
					putchar ('%');
					break;
				}
				
				if( *f == 'b')
				{
					if( argc > 0)
					{
						print_esc_string (*argv);
						++argv;
						--argc;
					}
					
					break;
				}
				
				if( strchr( "-+ #", *f))
				{
					++f;
					++direc_length;
				}
				
				if( *f == '*')
				{
					++f;
					++direc_length;
					
					if (argc > 0)
					{
						field_width = xstrtoul (*argv);
						++argv;
						--argc;
					}
					else
					{
						field_width = 0;
					}
				}
				else
				{
					while( isdigit (*f))
					{
						++f;
						++direc_length;
					}
					
					if( *f == '.')
					{
						++f;
						++direc_length;
						
						if( *f == '*')
						{
							++f;
							++direc_length;
							
							if (argc > 0)
							{
								precision = xstrtoul (*argv);
								++argv;
								--argc;
							}
							else
							{
								precision = 0;
							}
						}
						else
						{
							while( isdigit (*f))
							{
								++f;
								++direc_length;
							}
						}
					}
					
					if( *f == 'l' || *f == 'L' || *f == 'h')
					{
						++f;
						++direc_length;
					}
					
					if( !strchr( "diouxXfeEgGcs", *f))
					{
						error( 1, 0, "%%%c: invalid directive", *f);
					}
					
					++direc_length;
					
					if( argc > 0)
					{
						print_direc( direc_start, direc_length, field_width,
							precision, *argv);
						++argv;
						--argc;
					}
					else
					{
						print_direc( direc_start, direc_length, field_width,
							precision, "");
					}
				}
				
				break;
			}

			case '\\':
			{
			  f += print_esc (f);
			  break;
			}

			default:
			{
			  putchar( *f);
			}
		}
	}

  return( save_argc - argc);
}

/****************************************************************************
Desc:	Print a \ escape sequence starting at ESCSTART.
		Return the number of characters in the escape sequence
		besides the backslash.
****************************************************************************/
int print_esc(
	char *		escstart)
{
	register char *	p = escstart + 1;
	int 					esc_value = 0;		// Value of \nnn escape
	int 					esc_length;			// Length of \nnn escape

	// \0ooo and \xhhh escapes have maximum length of 3 chars.
  
	if (*p == 'x')
	{
		for( esc_length = 0, ++p;
			  esc_length < 3 && isxdigit (*p);
			  ++esc_length, ++p)
		{
			esc_value = esc_value * 16 + hextobin (*p);
		}
		
		if (esc_length == 0)
		{
			error (1, 0, "missing hexadecimal number in escape");
			putchar (esc_value);
		}
	}
	else if( isdigit( *p))
	{
		for( esc_length = 0;
			  esc_length < 3 && isodigit (*p);
			  ++esc_length, ++p)
		{
			esc_value = esc_value * 8 + octtobin (*p);
		}
		
		putchar (esc_value);
	}
	else if( strchr ("\"\\abcfnrtv", *p))
	{
		print_esc_char (*p++);
	}
	else
	{
		error( 1, 0, "\\%c: invalid escape", *p);
	}
	
	return( p - escstart - 1);
}

/****************************************************************************
Desc:	Output a single-character \ escape
****************************************************************************/
void print_esc_char(
	char		c)
{
	switch (c)
	{
		case 'a':			// Alert
		{
			putchar (7);
			break;
		}
		
		case 'b':			// Backspace
		{
			putchar (8);
			break;
		}
		
		case 'c':			// Cancel the rest of the output
		{
			exit (0);
			break;
		}
		
		case 'f':			// Form feed
		{
			putchar (12);
			break;
		}
		
		case 'n':			// New line
		{
			putchar (10);
			break;
		}
		
		case 'r':			// Carriage return
		{
			putchar (13);
			break;
		}
		
		case 't':			// Horizontal tab
		{
			putchar (9);
			break;
		}
		
		case 'v':			// Vertical tab
		{
			putchar (11);
			break;
		}
	 
		default:
		{
			putchar (c);
			break;
		}
	}
}

/****************************************************************************
Desc:	Print string STR, evaluating \ escapes
****************************************************************************/
void print_esc_string(
	char *		str)
{
	for( ; *str; str++)
	{
		if (*str == '\\')
		{
			str += print_esc (str);
		}
	 	else
		{
			putchar (*str);
		}
	}
}

/****************************************************************************
Desc:	Output a % directive.  START is the start of the directive,
		LENGTH is its length, and ARGUMENT is its argument.
		If FIELD_WIDTH or PRECISION is non-negative, they are args for
		'*' values in those fields
****************************************************************************/
void print_direc(
	char *		start,
	int			length,
	int			field_width,
	int			precision,
	char *		argument)
{
	char *		p;		// Null-terminated copy of % directive
	
	p = (char *)malloc( (unsigned)(length + 1));
	strncpy (p, start, length);
	p[length] = 0;
	
	switch( p[ length - 1])
	{
		case 'd':
		case 'i':
		{
			if (field_width < 0)
			{
				if (precision < 0)
				{
					printf (p, xstrtol (argument));
				}
				else
				{
					printf (p, precision, xstrtol (argument));
				}
			}
			else
			{
				if (precision < 0)
				{
					printf (p, field_width, xstrtol (argument));
				}
				else
				{
					printf (p, field_width, precision, xstrtol (argument));
				}
			}
			
			break;
		}
	
		 case 'o':
		 case 'u':
		 case 'x':
		 case 'X':
		 {
			if (field_width < 0)
			{
				if (precision < 0)
				{
					printf (p, xstrtoul (argument));
				}
				else
				{
					printf (p, precision, xstrtoul (argument));
				}
			}
			else
			{
				if (precision < 0)
				{
					printf (p, field_width, xstrtoul (argument));
				}
				else
				{
					printf (p, field_width, precision, xstrtoul (argument));
				}
			}
			
			break;
		 }
		 
		 case 'f':
		 case 'e':
		 case 'E':
		 case 'g':
		 case 'G':
		 {
			if( field_width < 0)
			{
				if (precision < 0)
				{
					printf (p, xstrtod (argument));
				}
				else
				{
					printf (p, precision, xstrtod (argument));
				}
			}
			else
			{
				if (precision < 0)
				{
					printf (p, field_width, xstrtod (argument));
				}
				else
				{
					printf (p, field_width, precision, xstrtod (argument));
				}
			}
			
			break;
		 }
		
		 case 'c':
		 {
			printf (p, *argument);
			break;
		 }
		
		 case 's':
		 {
			if( field_width < 0)
			{
				if (precision < 0)
				{
					printf (p, argument);
				}
				else
				{
					printf (p, precision, argument);
				}
			}
			else
			{
				if (precision < 0)
				{
					printf (p, field_width, argument);
				}
				else
				{
					printf (p, field_width, precision, argument);
				}
			}
			
			break;
		}
	}
	
	free( p);
}

/****************************************************************************
Desc:
****************************************************************************/
unsigned long xstrtoul(
	char *		s)
{
	char *			end;
	unsigned long	val;
	
	errno = 0;
	val = strtoul (s, &end, 0);
	verify (s, end);
	
	return( val);
}

/****************************************************************************
Desc:
****************************************************************************/
long xstrtol( 
	char *		s)
{
	char *		end;
	long 			val;
	
	errno = 0;
	val = strtol (s, &end, 0);
	verify( s, end);
	
	return( val);
}

/****************************************************************************
Desc:
****************************************************************************/
double xstrtod(
	char *		s)
{
	char *		end;
	double 		val;
	
	errno = 0;
	val = strtod (s, &end);
	verify (s, end);
	
	return( val);
}

/****************************************************************************
Desc:
****************************************************************************/
void verify( 
	char *		s,
	char *		end)
{
	if( *end)
	{
		if( s == end)
		{
			error (0, 0, "%s: expected a numeric value", s);
		}
		else
		{
			error (0, 0, "%s: value not completely converted", s);
		}
		
		exit_status = 1;
	 }
}

/****************************************************************************
Desc:
****************************************************************************/
void error( 
	int		status,
	int		errnum,
	char *	message, ...)
{
  va_list args;

  fflush (stdout);
  fprintf (stderr, "%s: ", program_name);

  va_start( args, message);
  vfprintf( stderr, message, args);
  va_end( args);
  
  if (errnum)
  {
    fprintf( stderr, ": %d", errnum);
  }
  
  putc ('\n', stderr);
  
  fflush (stderr);
  
  if (status)
  {
    exit( status);
  }
}
