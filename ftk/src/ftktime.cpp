//------------------------------------------------------------------------------
// Desc:	Date and time functions
// Tabs:	3
//
// Copyright (c) 1991-2000, 2002-2003, 2005-2007 Novell, Inc.
// All Rights Reserved.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; version 2.1
// of the License.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Library Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, contact Novell, Inc.
//
// To contact Novell about this file by physical or electronic mail, 
// you may find current contact information at www.novell.com.
//
// $Id$
//------------------------------------------------------------------------------

#include "ftksys.h"

#if defined( FLM_RING_ZERO_NLM)

	extern "C" void ConvertTicksToSeconds(
		LONG		ticks,
		LONG *	seconds,
		LONG *	tenthsOfSeconds);

	extern "C" void ConvertSecondsToTicks(
		LONG		seconds,
		LONG		tenthsOfSeconds,
		LONG *	ticks);
		
	FINLINE unsigned long time(
		unsigned long *	pulSeconds)
	{
		clockAndStatus clock;

		GetCurrentClock( &clock);

		if (pulSeconds)
		{
			*pulSeconds = (unsigned long)clock [0];
		}

		return ( (unsigned long)clock[0]);
	}
		
#endif

#define	BASEYR			1970				// all gmt calcs done since 1970
#define	SECONDSPERDAY	86400l			// 24 hours * 60 minutes * 60 seconds
#define	SECONDSPERHOUR	3600				// 60 minutes * 60 seconds
#define	DDAYSPERYEAR	365				// 365 days/year

static FLMUINT8 ui8NumDaysPerMonth[2][12] = 
{
	{ 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 },
	{ 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 }
};

static FLMUINT16 ui16NumDaysFromJan1st[2][12] = 
{
	{ 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334 },
	{ 0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335 }
};

static F_TMSTAMP maxdate =
{ 
	2106, 1, 6, 11, 0, 0, 0
};

static FLMUINT f_timeLeapYearsSince1970(
	FLMUINT16 		year);

/****************************************************************************
Desc:	Gets the number of seconds since 1980 or 1970.
****************************************************************************/
void FTKAPI f_timeGetSeconds(
	FLMUINT	*		puiSeconds)
{
#if defined( FLM_WIN)
	*puiSeconds = (FLMUINT) time( NULL);

#elif defined( FLM_UNIX) || defined( FLM_NLM)
	*puiSeconds = (FLMUINT) time( 0);

#else
	#error Platform not supported
#endif
}

/****************************************************************************
Desc:		Gets the time stamp from the system clock.
Notes:
****************************************************************************/
void FTKAPI f_timeGetTimeStamp(
	F_TMSTAMP *		pTimeStamp)
{
#if defined( FLM_WIN)

	SYSTEMTIME	rightnow;

	GetLocalTime( &rightnow );

	pTimeStamp->year  = rightnow.wYear;
	pTimeStamp->month = (FLMUINT8)(rightnow.wMonth - 1);
	pTimeStamp->day   = (FLMUINT8)rightnow.wDay;

	pTimeStamp->hour   = (FLMUINT8)rightnow.wHour;
	pTimeStamp->minute = (FLMUINT8)rightnow.wMinute;
	pTimeStamp->second = (FLMUINT8)rightnow.wSecond;
	pTimeStamp->hundredth = rightnow.wMilliseconds / 10;

#elif defined( FLM_RING_ZERO_NLM)

	f_timeSecondsToDate( (FLMUINT)
		time( NULL) - f_timeGetLocalOffset(), pTimeStamp);

#elif defined( FLM_UNIX) || defined( FLM_LIBC_NLM)

	time_t 		now;
	struct tm 	rightnow;

	now = time( (time_t *) 0 );
	(void)localtime_r( &now, &rightnow );

	pTimeStamp->year = rightnow.tm_year + 1900;
	pTimeStamp->month = rightnow.tm_mon;
	pTimeStamp->day = rightnow.tm_mday;
	pTimeStamp->hour = rightnow.tm_hour;
	pTimeStamp->minute = rightnow.tm_min;
	pTimeStamp->second = rightnow.tm_sec;
	pTimeStamp->hundredth = 0;
#else
	#error Platform not supported
#endif
}

/****************************************************************************
Desc:		Returns the local time bias in seconds
****************************************************************************/
FLMINT FTKAPI f_timeGetLocalOffset( void)
{
	FLMINT		iOffset = 0;

#if defined( FLM_WIN)
	TIME_ZONE_INFORMATION		tzInfo;
	DWORD								retVal;

	retVal = GetTimeZoneInformation( &tzInfo);

	if( retVal != TIME_ZONE_ID_UNKNOWN)
	{
		iOffset =
			(retVal == TIME_ZONE_ID_DAYLIGHT && tzInfo.DaylightDate.wMonth
				? tzInfo.Bias + tzInfo.DaylightBias
				: tzInfo.Bias) * 60;
	}

#elif defined( FLM_RING_ZERO_NLM)

	Synchronized_Clock_T    SynchronizedClock;

	f_memset( &SynchronizedClock, 0, sizeof( SynchronizedClock));
	GetSyncClockFields(
		SYNCCLOCK_DAYLIGHT_BIT | SYNCCLOCK_DAYLIGHT_OFFSET_BIT |
		SYNCCLOCK_DAYLIGHT_ON_OFF_BIT | SYNCCLOCK_TIMEZONE_OFFSET_BIT,
		&SynchronizedClock);

	iOffset = (FLMINT)SynchronizedClock.timezoneOffset;
	if( SynchronizedClock.daylight && SynchronizedClock.daylightOnOff)
	{
		iOffset += (FLMINT)SynchronizedClock.daylightOffset;
	}
	
#elif defined( FLM_UNIX) || defined( FLM_LIBC_NLM)
	time_t		gmtTime;
	time_t		localTime;
	struct tm	gmtTm;

	gmtTime = time( (time_t *)0);
	gmtime_r( &gmtTime, &gmtTm);
	localTime = mktime( &gmtTm);
	iOffset = (FLMINT)((FLMINT64)localTime - (FLMINT64)gmtTime);

#else
	#error Platform not supported
#endif

	return( iOffset);
}

/****************************************************************************
Desc:	Count the number of leap years from 1970 to given year.

In:	year - FLMUINT16 value containing the year
Out:	(None)
Ret:	Number of leap years since 1970
Notes:	According to the Gregorian calendar (which we currently use), the
		year is a leap year if it is divisible by 4, unless it is a century
		year, then it must be divisible by 400.
****************************************************************************/
static FLMUINT  f_timeLeapYearsSince1970(
	FLMUINT16	ui16Year)
{
	FLMUINT		uiTemp;

	/* first calculate # of leap years since 1600 */

	ui16Year -= 1601;								/* ui16Year = number of years since 1600*/
	uiTemp = (										/* Count leap years						*/
		(ui16Year / 4) -							/* Count potential leap years			*/
		(ui16Year / 100) +							/* Subtract out century years			*/
		(ui16Year / 400) +							/* Add back in quadricentenial years*/
		1											/* And don't forget to count 1600	*/
	);

	/* now subtract # of leap years between 1600 and 1970 */
	/* (the following becomes a constant at compile time) */

	uiTemp -= ((BASEYR-1600) / 4) - ((BASEYR-1600) / 100) + 1;
	return(uiTemp);
}

/****************************************************************************
Desc:		Convert from seconds to the F_TMSTAMP structure.
Notes:
****************************************************************************/
void FTKAPI f_timeSecondsToDate(
	FLMUINT 			uiSeconds,
	F_TMSTAMP *		date)
{
	FLMUINT			uiLeapYear;
	FLMUINT			uiMonth;
	FLMUINT			uiDaysInMonth;
	FLMUINT			uiDay;

	uiDay = uiSeconds / SECONDSPERDAY;					// # of days since 1970
	date->year = (FLMUINT16)((uiDay / DDAYSPERYEAR)	+ BASEYR);
	uiDay = uiDay % DDAYSPERYEAR;							// # of days into year

	// Check to see that the value for the current day is greater than the
	// number of leap years since 1970.  This is because we will be
	// subtracting the leap days from the current day and we don't want
	// the value for the day to go negative.

	while( uiDay < f_timeLeapYearsSince1970(date->year)) // if day < # of leap years
	{
		date->year--;											// decrement the year
		uiDay += DDAYSPERYEAR;								// adjust day by days/year
	}

	uiDay -= f_timeLeapYearsSince1970( date->year);	// subtract leap days
	uiLeapYear = f_timeIsLeapYear( date->year );		// set leap year flag

	// Find what our offset into the current month is.
	// To do this, we subtract out the number of days for each month, until
	// the number of days left does not span the end of the current month

	for( uiMonth = 0;
		  uiMonth < 12 &&
			(uiDay >= (uiDaysInMonth = ui8NumDaysPerMonth[uiLeapYear][uiMonth]));
		  uiMonth++)
	{
		uiDay -= uiDaysInMonth;								// subtract days in month
	}
	date->month = (FLMUINT8) uiMonth;					// set month, day
	date->day = (FLMUINT8) (++uiDay);

	uiDay = uiSeconds % SECONDSPERDAY;					// mod by seconds/day
	date->hour = (FLMUINT8)(uiDay / SECONDSPERHOUR);// get # of hours
	uiDay = uiDay % SECONDSPERHOUR;
	date->minute = (FLMUINT8)(uiDay / 60);				// get # of minutes
	date->second = (FLMUINT8)(uiDay % 60);
	date->hundredth = 0;										// no fractional seconds
}

/****************************************************************************
Desc:		Convert a time stamp to the number of seconds.
****************************************************************************/
void FTKAPI f_timeDateToSeconds(
	F_TMSTAMP *		pTimeStamp,
	FLMUINT *		puiSeconds)
{
	FLMUINT			uiDays = 0;

	// is date past max?
	
	if( f_timeCompareTimeStamps( pTimeStamp, &maxdate, 0) > 0)
	{
			*pTimeStamp = maxdate;
	}

	// Do date portion of calculation - result is days since 1/1/1970.

	if( pTimeStamp->year)
	{
		uiDays =
			(pTimeStamp->year - BASEYR) * 365 +	// years since BASE * days
			f_timeLeapYearsSince1970( pTimeStamp->year) +// leap years since BASE
			ui16NumDaysFromJan1st[ f_timeIsLeapYear(pTimeStamp->year)][pTimeStamp->month] +
			pTimeStamp->day - 1;						// days since 1st of month
	}

	//	Do time part of calculation - secs since 1/1/1970 12:00am.

	*puiSeconds = (((uiDays * 24) +				// convert days to hours
		pTimeStamp->hour ) * 60	+					// convert hours to min
		pTimeStamp->minute) * 60	+				// convert min to sec
		pTimeStamp->second;							// give secs granularity

}

/****************************************************************************
Desc:	Compare two time stamps
In:	date1, date2 - pointers to two DATIM structures
		flag - flag to indicate the type of comparison
			0 - compare date and time
			1 - compare date only
			2 - compare time only
Out:
Ret:	-1 if date1 is less than date2
		 0 if date1 is equal to date2
		 1 if date1 is greater than date2
Notes:
****************************************************************************/
FLMINT FTKAPI f_timeCompareTimeStamps(
	F_TMSTAMP *		pTimeStamp1,
	F_TMSTAMP *		pTimeStamp2,
	FLMUINT			flag)
{
	if( flag != 2)				/* not comparing times only	*/
	{
		if( pTimeStamp1->year != pTimeStamp2->year)
		{
			return((pTimeStamp1->year < pTimeStamp2->year) ? -1 : 1);
		}
		if( pTimeStamp1->month != pTimeStamp2->month)
		{
			return((pTimeStamp1->month < pTimeStamp2->month) ? -1 : 1);
		}
		if( pTimeStamp1->day != pTimeStamp2->day)
		{
			return((pTimeStamp1->day < pTimeStamp2->day) ? -1 : 1);
		}
	}
	if( flag != 1)
	{
		if( pTimeStamp1->hour != pTimeStamp2->hour)
		{
			return((pTimeStamp1->hour < pTimeStamp2->hour) ? -1 : 1);
		}
		if( pTimeStamp1->minute != pTimeStamp2->minute)
		{
			return((pTimeStamp1->minute < pTimeStamp2->minute) ? -1 : 1);
		}
		if( pTimeStamp1->second != pTimeStamp2->second)
		{
			return((pTimeStamp1->second < pTimeStamp2->second) ? -1 : 1);
		}
	}
	return( 0);
}

/****************************************************************************
Desc:		Get the current time in milliseconds.
****************************************************************************/
#if defined( FLM_UNIX) || defined( FLM_LIBC_NLM)
unsigned f_timeGetMilliTime()
{
#if defined( FLM_SOLARIS)
	return( (unsigned)((FLMUINT64)gethrtime() / (FLMUINT64)1000000));
#else
	struct timeval tv;
	
	gettimeofday(&tv, 0);

	return( (unsigned)((((FLMUINT64)tv.tv_sec * (FLMUINT64)1000000) +
		(FLMUINT64)tv.tv_usec) / 1000));
#endif
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
FLMUINT FTKAPI FLM_GET_TIMER( void)
{
#if defined( FLM_WIN)
	return( (FLMUINT)GetTickCount());
#elif defined( FLM_RING_ZERO_NLM)
	return( (FLMUINT)GetCurrentTime());
#else
	return( f_timeGetMilliTime());
#endif
}

/****************************************************************************
Desc:
****************************************************************************/
FLMUINT FTKAPI FLM_ELAPSED_TIME(
	FLMUINT			uiLaterTime,
	FLMUINT			uiEarlierTime)
{
	// This method of calculating elapsed time accounts for the
	// possibility of the time wrapping - which it will for some
	// of our counters (FLM_WIN is milliseconds and wraps in 49.7 days).

	return( (FLMUINT)(((uiLaterTime) >= (uiEarlierTime))
					? (FLMUINT)((uiLaterTime) - (uiEarlierTime))
					: (FLMUINT)((0xFFFFFFFF - (uiEarlierTime)) + (uiLaterTime))));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMUINT FTKAPI FLM_SECS_TO_TIMER_UNITS( 
	FLMUINT			uiSeconds)
{
#ifndef FLM_RING_ZERO_NLM
	return( uiSeconds * 1000);
#else
	LONG		uiTU;

	ConvertSecondsToTicks( (LONG)(uiSeconds), 0, &uiTU);
	return( (FLMUINT)uiTU);
#endif
}

/****************************************************************************
Desc:
****************************************************************************/
FLMUINT FTKAPI FLM_TIMER_UNITS_TO_SECS( 
	FLMUINT			uiTU)
{
#ifndef FLM_RING_ZERO_NLM
	return( uiTU / 1000);
#else
	LONG	uiSeconds;
	LONG	udDummy;
	
	ConvertTicksToSeconds( (LONG)uiTU, &uiSeconds, &udDummy);
	return( (FLMUINT)uiSeconds);
#endif
}

/****************************************************************************
Desc:
****************************************************************************/
FLMUINT FTKAPI FLM_TIMER_UNITS_TO_MILLI( 
	FLMUINT			uiTU)
{
#ifndef FLM_RING_ZERO_NLM
	return( uiTU);
#else
	LONG		udTenths;
	LONG		udSeconds;
	
	ConvertTicksToSeconds( (LONG)uiTU, &udSeconds, &udTenths);
	return( (FLMUINT)(udSeconds) * 1000 + (FLMUINT)udTenths * 100);
#endif
}
	
/****************************************************************************
Desc:
****************************************************************************/
FLMUINT FTKAPI FLM_MILLI_TO_TIMER_UNITS( 
	FLMUINT			uiMilliSeconds)
{
#ifndef FLM_RING_ZERO_NLM
	return( uiMilliSeconds);
#else
	LONG 		udTenths;
	LONG 		udSeconds;
	LONG		uiTU;
	
	udSeconds = ((LONG) uiMilliSeconds) / 1000;
	udTenths = (((LONG) uiMilliSeconds) % 1000) / 100;
	
	ConvertSecondsToTicks( udSeconds, udTenths, &uiTU);
	return( uiTU);
#endif
}

/********************************************************************
Desc: Determine if a given year is a leap year.
*********************************************************************/
FINLINE FLMUINT f_leapYear(
	FLMUINT		uiYear)
{
	if (uiYear % 4 != 0)
	{
		return( 0);
	}
	
	if (uiYear % 100 != 0)
	{
		return( 1);
	}
	
	if (uiYear % 400 != 0)
	{
		return( 0);
	}
	
	return( 1);
}

/********************************************************************
Desc: Calculate days in a given month of a given year.
*********************************************************************/
FSTATIC FLMUINT f_daysInMonth(
	FLMUINT		uiYear,
	FLMUINT		uiMonth)
{
	switch (uiMonth + 1)
	{
		case 4:
		case 6:
		case 9:
		case 11:
			return( 30);
		case 2:
			return( 28 + f_leapYear( uiYear));
		default:
			return( 31);
	}
}

/********************************************************************
Desc:
*********************************************************************/
FSTATIC void f_adjustTime(
	F_TMSTAMP *	pTime,
	FLMINT		iStartPoint)
{
	switch (iStartPoint)
	{
		case 1:
			goto Adj_1;
		case 2:
			goto Adj_2;
		case 3:
			goto Adj_3;
		case 4:
			goto Adj_4;
		case 5:
			goto Adj_5;
		case 6:
			goto Adj_6;
	}
Adj_1:
	if (pTime->hundredth >= 100)
	{
		pTime->second++;
		pTime->hundredth = 0;
	}
Adj_2:
	if (pTime->second == 60)
	{
		pTime->minute++;
		pTime->second = 0;
	}
Adj_3:
	if (pTime->minute == 60)
	{
		pTime->hour++;
		pTime->minute = 0;
	}
Adj_4:
	if (pTime->hour == 24)
	{
		pTime->day++;
		pTime->hour = 0;
	}
Adj_5:
	if ((FLMUINT)pTime->day > f_daysInMonth( pTime->year, pTime->month))
	{
		pTime->month++;
		pTime->day = 1;
	}
Adj_6:
	if (pTime->month > 11)
	{
		pTime->year++;
		pTime->month = 1;
	}
}

/****************************************************************************
Desc: Calculate the elapsed time, including milliseconds.
****************************************************************************/
void FTKAPI f_addElapsedTime(
	F_TMSTAMP *	pStartTime,
	FLMUINT64 *	pui64ElapMilli)
{
	F_TMSTAMP	StartTime;
	F_TMSTAMP	EndTime;
	FLMUINT		uiSec = 0;
	FLMUINT		uiHundredth = 0;

	f_timeGetTimeStamp( &EndTime);
	f_memcpy( &StartTime, pStartTime, sizeof( F_TMSTAMP));

	if (StartTime.year < EndTime.year)
	{
		if (StartTime.hundredth)
		{
			uiHundredth += (FLMUINT)(100 - StartTime.hundredth);
			StartTime.hundredth = 0;
			StartTime.second++;
			f_adjustTime( &StartTime, 2);
		}
		if (StartTime.second)
		{
			uiSec += (FLMUINT)(60 - StartTime.second);
			StartTime.second = 0;
			StartTime.minute++;
			f_adjustTime( &StartTime, 3);
		}
		if (StartTime.minute)
		{
			uiSec += (FLMUINT)((60 - StartTime.minute) * 60);
			StartTime.minute = 0;
			StartTime.hour++;
			f_adjustTime( &StartTime, 4);
		}
		if (StartTime.hour)
		{
			uiSec += (FLMUINT)((24 - StartTime.hour) * 3600);
			StartTime.hour = 0;
			StartTime.day++;
			f_adjustTime( &StartTime, 5);
		}
		if (StartTime.day > 1)
		{
			uiSec += (FLMUINT)(f_daysInMonth( StartTime.year, StartTime.month) -
									StartTime.day + 1) * (FLMUINT)86400;
			StartTime.day = 1;
			StartTime.month++;
			f_adjustTime( &StartTime, 6);
		}
		if (StartTime.month > 1)
		{
			while (StartTime.month <= 11)
			{
				uiSec += (FLMUINT)((FLMUINT)f_daysInMonth( StartTime.year,
										StartTime.month) * (FLMUINT)86400);
				StartTime.month++;
			}
			StartTime.year++;
		}
		while (StartTime.year < EndTime.year)
		{
			uiSec += (FLMUINT)((FLMUINT)(365 + f_leapYear( StartTime.year)) *
							(FLMUINT)86400);
			StartTime.year++;
		}
	}

	if (StartTime.month < EndTime.month)
	{
		if (StartTime.hundredth)
		{
			uiHundredth += (FLMUINT)(100 - StartTime.hundredth);
			StartTime.hundredth = 0;
			StartTime.second++;
			f_adjustTime( &StartTime, 2);
		}
		if (StartTime.second)
		{
			uiSec += (FLMUINT)(60 - StartTime.second);
			StartTime.second = 0;
			StartTime.minute++;
			f_adjustTime( &StartTime, 3);
		}
		if (StartTime.minute)
		{
			uiSec += (FLMUINT)((60 - StartTime.minute) * 60);
			StartTime.minute = 0;
			StartTime.hour++;
			f_adjustTime( &StartTime, 4);
		}
		if (StartTime.hour)
		{
			uiSec += (FLMUINT)((24 - StartTime.hour) * 3600);
			StartTime.hour = 0;
			StartTime.day++;
			f_adjustTime( &StartTime, 5);
		}
		if (StartTime.day > 1)
		{
			uiSec += (FLMUINT)(f_daysInMonth( StartTime.year, StartTime.month) -
									StartTime.day + 1) * (FLMUINT)86400;
			StartTime.day = 1;
			StartTime.month++;
			f_adjustTime( &StartTime, 6);
		}
		while (StartTime.month < EndTime.month)
		{
			uiSec += (FLMUINT)((FLMUINT)f_daysInMonth( StartTime.year,
									StartTime.month) * (FLMUINT)86400);
			StartTime.month++;
		}
	}

	if (StartTime.day < EndTime.day)
	{
		if (StartTime.hundredth)
		{
			uiHundredth += (FLMUINT)(100 - StartTime.hundredth);
			StartTime.hundredth = 0;
			StartTime.second++;
			f_adjustTime( &StartTime, 2);
		}
		if (StartTime.second)
		{
			uiSec += (FLMUINT)(60 - StartTime.second);
			StartTime.second = 0;
			StartTime.minute++;
			f_adjustTime( &StartTime, 3);
		}
		if (StartTime.minute)
		{
			uiSec += (FLMUINT)((60 - StartTime.minute) * 60);
			StartTime.minute = 0;
			StartTime.hour++;
			f_adjustTime( &StartTime, 4);
		}
		if (StartTime.hour)
		{
			uiSec += (FLMUINT)((24 - StartTime.hour) * 3600);
			StartTime.hour = 0;
			StartTime.day++;
			f_adjustTime( &StartTime, 5);
		}
		uiSec += (FLMUINT)(EndTime.day - StartTime.day) * (FLMUINT)86400;
		StartTime.day = 1;
		StartTime.month++;
		f_adjustTime( &StartTime, 6);
	}

	if (StartTime.hour < EndTime.hour)
	{
		if (StartTime.hundredth)
		{
			uiHundredth += (FLMUINT)(100 - StartTime.hundredth);
			StartTime.hundredth = 0;
			StartTime.second++;
			f_adjustTime( &StartTime, 2);
		}
		if (StartTime.second)
		{
			uiSec += (FLMUINT)(60 - StartTime.second);
			StartTime.second = 0;
			StartTime.minute++;
			f_adjustTime( &StartTime, 3);
		}
		if (StartTime.minute)
		{
			uiSec += (FLMUINT)((60 - StartTime.minute) * 60);
			StartTime.minute = 0;
			StartTime.hour++;
			f_adjustTime( &StartTime, 4);
		}
		uiSec += (FLMUINT)((EndTime.hour - StartTime.hour) * 3600);
		StartTime.hour = 0;
		StartTime.day++;
		f_adjustTime( &StartTime, 5);
	}

	if (StartTime.minute < EndTime.minute)
	{
		if (StartTime.hundredth)
		{
			uiHundredth += (FLMUINT)(100 - StartTime.hundredth);
			StartTime.hundredth = 0;
			StartTime.second++;
			f_adjustTime( &StartTime, 2);
		}
		if (StartTime.second)
		{
			uiSec += (FLMUINT)(60 - StartTime.second);
			StartTime.second = 0;
			StartTime.minute++;
			f_adjustTime( &StartTime, 3);
		}
		uiSec += (FLMUINT)((EndTime.minute - StartTime.minute) * 60);
		StartTime.minute = 0;
		StartTime.hour++;
		f_adjustTime( &StartTime, 4);
	}

	if (StartTime.second < EndTime.second)
	{
		if (StartTime.hundredth)
		{
			uiHundredth += (FLMUINT)(100 - StartTime.hundredth);
			StartTime.hundredth = 0;
			StartTime.second++;
			f_adjustTime( &StartTime, 2);
		}
		uiSec += (FLMUINT)(EndTime.second - StartTime.second);
		StartTime.second = 0;
		StartTime.minute++;
		f_adjustTime( &StartTime, 3);
	}

	if (StartTime.hundredth < EndTime.hundredth)
	{
		uiHundredth += (FLMUINT)(EndTime.hundredth - StartTime.hundredth);
	}
	if (uiSec)
	{
		(*pui64ElapMilli) += (FLMUINT64)((uiHundredth * 10 + uiSec * 1000));
	}
	else
	{
		(*pui64ElapMilli) += (FLMUINT64)(uiHundredth * 10);
	}
}
