#ifndef ARIA_PTHREAD_COMPAT_H
#define ARIA_PTHREAD_COMPAT_H
#ifdef _WIN32
#include <windows.h>
#include <process.h>
typedef CRITICAL_SECTION pthread_mutex_t;
typedef CONDITION_VARIABLE pthread_cond_t;
typedef HANDLE pthread_t;
#define PTHREAD_MUTEX_INITIALIZER {0}
#define PTHREAD_COND_INITIALIZER {0}
static inline int pthread_mutex_lock(pthread_mutex_t *m){static int i; if(!i){InitializeCriticalSection(m);i=1;} EnterCriticalSection(m);return 0;}
static inline int pthread_mutex_unlock(pthread_mutex_t *m){LeaveCriticalSection(m);return 0;}
static inline int pthread_cond_wait(pthread_cond_t *c,pthread_mutex_t*m){SleepConditionVariableCS(c,m,INFINITE);return 0;}
static inline int pthread_cond_signal(pthread_cond_t *c){WakeConditionVariable(c);return 0;}
typedef unsigned (__stdcall *aria_thread_fn)(void*);
static inline int pthread_create(pthread_t*t,void*a,void*(*fn)(void*),void*x){(void)a; *t=(HANDLE)_beginthreadex(NULL,0,(aria_thread_fn)fn,x,0,NULL);return *t?0:-1;}
static inline int pthread_join(pthread_t t,void**r){(void)r; WaitForSingleObject(t,INFINITE); CloseHandle(t);return 0;}
#endif
#endif
