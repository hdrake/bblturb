C $Header: /u/gcmpack/MITgcm/pkg/mypackage/mypackage_tendency_apply.F,v 1.4 2014/07/09 17:00:49 jmc Exp $
C $Name:  $

#include "MYPACKAGE_OPTIONS.h"
C--  File mypackage_tendency_apply.F: Routines to apply MYPACKAGE tendencies
C--   Contents
C--   o MYPACKAGE_TENDENCY_APPLY_U
C--   o MYPACKAGE_TENDENCY_APPLY_V
C--   o MYPACKAGE_TENDENCY_APPLY_T
C--   o MYPACKAGE_TENDENCY_APPLY_S

C---+----1----+----2----+----3----+----4----+----5----+----6----+----7-|--+----|
CBOP 0
C !ROUTINE: MYPACKAGE_TENDENCY_APPLY_U

C !INTERFACE:
      SUBROUTINE MYPACKAGE_TENDENCY_APPLY_U(
     U                     gU_arr,
     I                     iMin,iMax,jMin,jMax, k, bi, bj,
     I                     myTime, myIter, myThid )

C     !DESCRIPTION:
C     Add mypackage tendency terms to U tendency.  Routine works for one
C     level at a time. Assumes that U and V tendencies are on C-Grid or
C     on A-grid.

C     !USES:
      IMPLICIT NONE
#include "SIZE.h"
#include "EEPARAMS.h"
#include "GRID.h"
#include "PARAMS.h"
#include "DYNVARS.h"
#include "MYPACKAGE.h"
#include "FFIELDS.h"

C     !INPUT/OUTPUT PARAMETERS:
C     gU_arr    :: the tendency array
C     iMin,iMax :: Working range of x-index for applying forcing.
C     jMin,jMax :: Working range of y-index for applying forcing.
C     k         :: Current vertical level index
C     bi,bj     :: Current tile indices
C     myTime    :: Current time in simulation
C     myIter    :: Current iteration number
C     myThid    :: my Thread Id number
      _RL     gU_arr(1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      INTEGER iMin, iMax, jMin, jMax
      INTEGER k, bi, bj
      _RL     myTime
      INTEGER myIter
      INTEGER myThid
CEOP

#ifdef MYPACKAGE_TENDENCY
C     !LOCAL VARIABLES:
      INTEGER i, j

C     ================================================================
C     Edited by Henri F. Drake, 02/14/2019

      REAL*8 tauTop
      REAL*8 tauProfile

      REAL*8 U0force
      REAL*8 U0relax
      REAL*8 omega_tide
      REAL*8 phi_tide

C     Parameters for approximate Lunar Diurnal tide
      phi_tide = 2.1268284479803232
      omega_tide  = 2*3.14159265359/(12*3600)
      U0relax = 0.1182029036394405/3.
      U0force = U0relax / ((omega_tide*omega_tide) /
     &             (omega_tide*omega_tide-f0*f0))

C     Parameters for relaxation layer
      tauTop = 1./(3600.)
      tauProfile = 0.
      if((rC(k)-rC(Nr)).gt.1600.) then
      tauProfile=tauTop*((rC(k)-rC(Nr))-1600.)/((rC(1)-rC(Nr))-1600.)
      end if

      IF ( myPa_applyTendU .AND. myPa_Tend_Cgrid ) THEN
        DO j=jMin,jMax
         DO i=iMin,iMax
          gU_arr(i,j) = gU_arr(i,j)
     &     + maskW(i,j,k,bi,bj) * (
C     Add projection of buoyancy production term to u-tendency
     &      (-gravity
     &        *0.5*(rhoInSitu(i-1,j,k,bi,bj)+rhoInSitu(i,j,k,bi,bj))
     &        *recip_rhoConst
     &        *sin(slopeAngle)
     &      )
C     Add tidal forcing
     &      + U0force*omega_tide*cos(omega_tide*myTime)*cos(phi_tide)
C     Add tidal relaxation in sponge layer (what does this do?)
     &      + ( U0relax * ( cos(phi_tide)*sin(omega_tide*myTime)
     &                     -sin(phi_tide)*cos(omega_tide*myTime)
     &                     *f0/omega_tide )
     &         -uVel(i,j,k,bi,bj) )
     &      * tauProfile
     &     )

         ENDDO
        ENDDO
      ELSEIF ( myPa_applyTendU ) THEN
C--   Assume Agrid position:
        DO j=jMin,jMax
         DO i=iMin,iMax
          gU_arr(i,j) = gU_arr(i,j)
     &     + maskW(i,j,k,bi,bj)*( myPa_TendVelU(i-1,j,k,bi,bj)
     &                           +myPa_TendVelU( i ,j,k,bi,bj)
     &                          )*0.5 _d 0
         ENDDO
        ENDDO
      ENDIF

#endif /* MYPACKAGE_TENDENCY */

      RETURN
      END

C---+----1----+----2----+----3----+----4----+----5----+----6----+----7-|--+----|
CBOP 0
C !ROUTINE: MYPACKAGE_TENDENCY_APPLY_V

C !INTERFACE:
      SUBROUTINE MYPACKAGE_TENDENCY_APPLY_V(
     U                     gV_arr,
     I                     iMin,iMax,jMin,jMax, k, bi, bj,
     I                     myTime, myIter, myThid )

C     !DESCRIPTION:
C     Add mypackage tendency terms to V tendency.  Routine works for one
C     level at a time. Assumes that U and V tendencies are on C-Grid or
C     on A-grid.

C     !USES:
      IMPLICIT NONE
#include "SIZE.h"
#include "EEPARAMS.h"
#include "GRID.h"
#include "PARAMS.h"
#include "DYNVARS.h"
#include "MYPACKAGE.h"
#include "FFIELDS.h"

C     !INPUT/OUTPUT PARAMETERS:
C     gV_arr    :: the tendency array
C     iMin,iMax :: Working range of x-index for applying forcing.
C     jMin,jMax :: Working range of y-index for applying forcing.
C     k         :: Current vertical level index
C     bi,bj     :: Current tile indices
C     myTime    :: Current time in simulation
C     myIter    :: Current iteration number
C     myThid    :: my Thread Id number
      _RL     gV_arr(1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      INTEGER iMin, iMax, jMin, jMax
      INTEGER k, bi, bj
      _RL     myTime
      INTEGER myIter
      INTEGER myThid
CEOP

#ifdef MYPACKAGE_TENDENCY
C     !LOCAL VARIABLES:
      INTEGER i, j, kp1

C     ================================================================
C     Edited by Henri F. Drake, 02/14/2019
      REAL*8 tauTop
      REAL*8 tauProfile

      REAL*8 U0force
      REAL*8 U0relax
      REAL*8 omega_tide
      REAL*8 phi_tide

C     Parameters for approximate Lunar Diurnal tide
      phi_tide = 2.1268284479803232
      omega_tide  = 2*3.14159265359/(12*3600)
      U0relax = 0.1182029036394405/3.
      U0force = U0relax / ((omega_tide*omega_tide) /
     &             (omega_tide*omega_tide-f0*f0))

C     Parameters for relaxation layer
      tauTop = 1./(3600.)
      tauProfile = 0.
      if((rC(k)-rC(Nr)).gt.1600.) then
      tauProfile=tauTop*((rC(k)-rC(Nr))-1600.)/((rC(1)-rC(Nr))-1600.)
      end if

      IF ( myPa_applyTendV .AND. myPa_Tend_Cgrid ) THEN
        kp1=max(k+1,Nr)
        DO j=jMin,jMax
         DO i=iMin,iMax
          gV_arr(i,j) = gV_arr(i,j)
     &     + maskS(i,j,k,bi,bj) * (
C     Add tidal forcing
     &      + U0force*omega_tide*cos(omega_tide*myTime)*sin(phi_tide)
C     Add tidal relaxation in sponge layer (what does this do?)
     &      + ( U0relax * ( sin(phi_tide)*sin(omega_tide*myTime)
     &                     +cos(phi_tide)*cos(omega_tide*myTime)
     &                     *f0/omega_tide )
     &         -vVel(i,j,k,bi,bj) )
     &      * tauProfile
     &     )
         ENDDO
        ENDDO
C     ================================================================

      ELSEIF ( myPa_applyTendV ) THEN
C--   Assume Agrid position:
        DO j=jMin,jMax
         DO i=iMin,iMax
          gV_arr(i,j) = gV_arr(i,j)
     &     + maskS(i,j,k,bi,bj)*( myPa_TendVelV(i,j-1,k,bi,bj)
     &                           +myPa_TendVelV(i, j ,k,bi,bj)
     &                          )*0.5 _d 0
         ENDDO
        ENDDO
      ENDIF

#endif /* MYPACKAGE_TENDENCY */

      RETURN
      END

C---+----1----+----2----+----3----+----4----+----5----+----6----+----7-|--+----|
CBOP 0
C !ROUTINE: MYPACKAGE_TENDENCY_APPLY_T

C !INTERFACE:
      SUBROUTINE MYPACKAGE_TENDENCY_APPLY_T(
     U                     gT_arr,
     I                     iMin,iMax,jMin,jMax, k, bi, bj,
     I                     myTime, myIter, myThid )

C     !DESCRIPTION:
C     Add mypackage tendency terms to T (theta) tendency.  Routine works
C     for one level at a time

C     !USES:
      IMPLICIT NONE
#include "SIZE.h"
#include "EEPARAMS.h"
#include "GRID.h"
#include "PARAMS.h"
#include "DYNVARS.h"
#include "MYPACKAGE.h"
#include "FFIELDS.h"

C     !INPUT/OUTPUT PARAMETERS:
C     gT_arr    :: the tendency array
C     iMin,iMax :: Working range of x-index for applying forcing.
C     jMin,jMax :: Working range of y-index for applying forcing.
C     k         :: Current vertical level index
C     bi,bj     :: Current tile indices
C     myTime    :: Current time in simulation
C     myIter    :: Current iteration number
C     myThid    :: my Thread Id number
      _RL     gT_arr(1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL     maskDown(1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL     maskUp(1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      INTEGER iMin, iMax, jMin, jMax
      INTEGER k, bi, bj
      _RL     myTime
      INTEGER myIter
      INTEGER myThid
CEOP
#ifdef MYPACKAGE_TENDENCY
C     !LOCAL VARIABLES:
      INTEGER i, j, kp1, km1

C     ================================================================
C     Edited by Henri F. Drake, 01/31/2019

      REAL*8 tauTop
      REAL*8 tauProfile

C     Parameters for relaxation layer
      tauTop = 1./(3600.)
      tauProfile = 0.
      if((rC(k)-rC(Nr)).gt.1600.) then
      tauProfile=tauTop*((rC(k)-rC(Nr))-1600.)/((rC(1)-rC(Nr))-1600.)
      end if

C     Apply tendencies on temperature perturbations due to advection of
C     background stratification by vertical velocity perturbation
C     and differential diffusion of background stratification by
C     a vertically-non-uniform mixing coefficient,
C     as well as a sponge layer to absorb internal waves radiated
C     from the barotropic tide (see above) sloshing over topography
      IF ( myPa_applyTendT ) THEN
        kp1=min(k+1,Nr)
        km1=max(1,k-1)
        DO j=jMin,jMax
         DO i=iMin,iMax
          maskUp(i,j) = maskC(i,j,km1,bi,bj)*maskC(i,j,k,bi,bj)
          maskDown(i,j) = maskC(i,j,k,bi,bj)*maskC(i,j,kp1,bi,bj)
          gT_arr(i,j) = gT_arr(i,j)
C     Advection by true vertical velocity (with cross-slope and
C     slope-normal components)
     &       - (0.5 * (wVel(i,j,k,bi,bj)*maskUp(i,j) +
     &                 wVel(i,j,kp1,bi,bj)*maskDown(i,j))
     &              * cos(slopeAngle)
     &         +0.5 * (uVel(i,j,k,bi,bj)*maskW(i,j,k,bi,bj) +
     &                 uVel(i+1,j,k,bi,bj)*maskW(i+1,j,k,bi,bj))
     &              * sin(slopeAngle))
     &       * lapseRate
#ifdef ALLOW_3D_DIFFKR
C     Differential diffusion by non-constant true-vertical imposed diffusivity
     &       + (  diffKr(i,j,k,bi,bj)*maskUp(i,j)
     &          - diffKr(i,j,kp1,bi,bj)*maskDown(i,j))
     &       * cos(slopeAngle) * lapseRate
     &       * recip_drF(k) * _recip_hFacC(i,j,k,bi,bj)
#endif /* ALLOW_3D_DIFFKR */
C     Add sponge layer to absorb internal waves
     &       + ( Trlx(k) - theta(i,j,k,bi,bj) ) * tauProfile
         ENDDO
        ENDDO
      ENDIF
C     ==============================================================

#endif /* MYPACKAGE_TENDENCY */

      RETURN
      END

C---+----1----+----2----+----3----+----4----+----5----+----6----+----7-|--+----|
CBOP 0
C !ROUTINE: MYPACKAGE_TENDENCY_APPLY_S

C !INTERFACE:
      SUBROUTINE MYPACKAGE_TENDENCY_APPLY_S(
     U                     gS_arr,
     I                     iMin,iMax,jMin,jMax, k, bi, bj,
     I                     myTime, myIter, myThid )

C     !DESCRIPTION:
C     Add mypackage tendency terms to S tendency.  Routine works for one
C     level at a time

C     !USES:
      IMPLICIT NONE
#include "SIZE.h"
#include "EEPARAMS.h"
#include "GRID.h"
c#include "DYNVARS.h"
#include "MYPACKAGE.h"

C     !INPUT/OUTPUT PARAMETERS:
C     gS_arr    :: the tendency array
C     iMin,iMax :: Working range of x-index for applying forcing.
C     jMin,jMax :: Working range of y-index for applying forcing.
C     k         :: Current vertical level index
C     bi,bj     :: Current tile indices
C     myTime    :: Current time in simulation
C     myIter    :: Current iteration number
C     myThid    :: my Thread Id number
      _RL     gS_arr(1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      INTEGER iMin, iMax, jMin, jMax
      INTEGER k, bi, bj
      _RL     myTime
      INTEGER myIter
      INTEGER myThid
CEOP

#ifdef MYPACKAGE_TENDENCY
C     !LOCAL VARIABLES:
      INTEGER i, j

      IF ( myPa_applyTendS ) THEN
        DO j=jMin,jMax
         DO i=iMin,iMax
          gS_arr(i,j) = gS_arr(i,j)
     &     + maskC(i,j,k,bi,bj) * myPa_TendScal2(i,j,k,bi,bj)
         ENDDO
        ENDDO
      ENDIF

#endif /* MYPACKAGE_TENDENCY */

      RETURN
      END
