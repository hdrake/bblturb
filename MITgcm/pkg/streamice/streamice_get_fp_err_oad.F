C $Header: /u/gcmpack/MITgcm/pkg/streamice/streamice_get_fp_err_oad.F,v 1.3 2016/03/18 19:18:29 dgoldberg Exp $
C $Name:  $

#include "STREAMICE_OPTIONS.h"

C---+----1----+----2----+----3----+----4----+----5----+----6----+----7-|--+----|

CBOP
      SUBROUTINE STREAMICE_GET_FP_ERR_OAD ( err_max, myThid )
C     /============================================================\
C     | SUBROUTINE                                                 |
C     | o                                                          |
C     |============================================================|
C     |                                                            |
C     \============================================================/
      IMPLICIT NONE

C     === Global variables ===
#include "SIZE.h"
#include "EEPARAMS.h"
#include "PARAMS.h"
#include "STREAMICE.h"
#include "STREAMICE_CG.h"


C     !INPUT/OUTPUT ARGUMENTS
      _RL err_max
      INTEGER myThid

#ifdef ALLOW_STREAMICE
#if (defined (ALLOW_STREAMICE_OAD_FP))
      INTEGER conv_flag, i, j, bi, bj

      INTEGER ikey_nl
      _RL err_tempu, err_tempv
      _RL max_vel, tempu, tempv, err_lastchange, cgtol
      CHARACTER*(MAX_LEN_MBUF) msgBuf
      _RL err_sum, conj_norm
      _RL err_sum_tile (nSx,nSy)

       err_max = 0. _d 0
       err_sum = 0. _d 0
       conj_norm = 0. _d 0 

       DO bj = myByLo(myThid), myByHi(myThid)
        DO bi = myBxLo(myThid), myBxHi(myThid)
         err_sum_tile(bi,bj) = 0. _d 0  
        ENDDO
       ENDDO
       


       DO bj = myByLo(myThid), myByHi(myThid)
        DO bi = myBxLo(myThid), myBxHi(myThid)
         DO j=1,sNy
          DO i=1,sNx
           err_tempu = 0. _d 0
           err_tempv = 0. _d 0
           IF (STREAMICE_umask(i,j,bi,bj).eq.1) THEN
            err_tempu =
     &       ABS (U_streamice(i,j,bi,bj)-u_new_SI(i,j,bi,bj))
           ENDIF
           IF (STREAMICE_vmask(i,j,bi,bj).eq.1) THEN
            err_tempv = MAX( err_tempu,
     &       ABS (V_streamice(i,j,bi,bj)-v_new_SI(i,j,bi,bj)))
           ENDIF
           IF (err_tempv .ge. err_max) err_max = err_tempv
          ENDDO
         ENDDO
        ENDDO
       ENDDO

       CALL GLOBAL_MAX_R8 (err_max, myThid)

#endif
#endif
      RETURN
      END
