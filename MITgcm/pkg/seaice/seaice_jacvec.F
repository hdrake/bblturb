C $Header: /u/gcmpack/MITgcm/pkg/seaice/seaice_jacvec.F,v 1.7 2016/04/22 08:50:34 mlosch Exp $
C $Name:  $

#include "SEAICE_OPTIONS.h"

CBOP
C     !ROUTINE: SEAICE_JACVEC
C     !INTERFACE:
      SUBROUTINE SEAICE_JACVEC( 
     I     uIceLoc, vIceLoc, uIceRes, vIceRes,
     U     duIce, dvIce,  
     I     newtonIter, krylovIter, myTime, myIter, myThid )

C     !DESCRIPTION: \bv
C     *==========================================================*
C     | SUBROUTINE SEAICE_JACVEC
C     | o For Jacobian-free Newton-Krylov solver compute
C     |   Jacobian times vector by finite difference approximation
C     *==========================================================*
C     | written by Martin Losch, Oct 2012
C     *==========================================================*
C     \ev

C     !USES:
      IMPLICIT NONE

C     === Global variables ===
#include "SIZE.h"
#include "EEPARAMS.h"
#include "PARAMS.h"
#include "DYNVARS.h"
#include "GRID.h"
#include "SEAICE_SIZE.h"
#include "SEAICE_PARAMS.h"
#include "SEAICE.h"

C     !INPUT/OUTPUT PARAMETERS:
C     === Routine arguments ===
C     myTime :: Simulation time
C     myIter :: Simulation timestep number
C     myThid :: my Thread Id. number
C     newtonIter :: current iterate of Newton iteration
C     krylovIter :: current iterate of Krylov iteration
      _RL     myTime
      INTEGER myIter
      INTEGER myThid
      INTEGER newtonIter
      INTEGER krylovIter
C     u/vIceLoc :: local copies of the current ice velocity
      _RL uIceLoc(1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy)
      _RL vIceLoc(1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy)
C     u/vIceRes :: initial residual of this Newton iterate
      _RL uIceRes(1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy)
      _RL vIceRes(1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy)
C     du/vIce   :: correction of ice velocities
      _RL duIce  (1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy)
      _RL dvIce  (1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy)

#ifdef SEAICE_ALLOW_JFNK
C     Local variables:
      _RL utp     (1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy)
      _RL vtp     (1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy)
C     u/vIceResP :: residual computed with u/vtp
      _RL uIceResP(1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy)
      _RL vIceResP(1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy)

C     i,j,bi,bj :: loop indices
      INTEGER i,j,bi,bj
      _RL epsilon, reps
CEOP
C     Instructions for using TAF or TAMC to generate exact Jacobian times
C     vector operations:
C
C     1. make small_f
C     2. cat seaice_calc_residual.f seaice_oceandrag_coeffs.f \
C        seaice_bottomdrag_coeffs.f \
C        seaice_calc_strainrates.f seaice_calc_viscosities.f \
C        seaice_calc_rhs.f seaice_calc_lhs.f > taf_input.f
C     3. staf -v1 -forward -toplevel seaice_calc_residual \
C             -input uIceLoc,viceLoc -output uIceRes,vIceRes taf_input.f
C     4. insert content of taf_input_ftl.f at the end of this file
C     5. add the following code and comment out the finite difference code
C
C     Instruction for using TAF 2.4 and higher (or staf with default -v2
C     starting with version 2.0):
C
C     1. make small_f
C     2. files="seaice_calc_residual.f seaice_oceandrag_coeffs.f \
C               seaice_bottomdrag_coeffs.f \
C               seaice_calc_strainrates.f seaice_calc_viscosities.f \
C               seaice_calc_rhs.f seaice_calc_lhs.f"
C     3. staf -forward -toplevel seaice_calc_residual \
C             -input uIceLoc,viceLoc -output uIceRes,vIceRes $files
C     4. copy files seaice_*_tl.f to the corresponding seaice_*.f files, 
C        e.g. with this bash script:
C     for file in $files; do 
C       nfile=`echo $file | awk -F. '{printf "%s_tl.f", $1}'`; 
C       \cp -f $nfile $file
C     done
C     5. add the following code, change "call g_seaice_calc_residual" 
C        to "call seaice_calc_residual_tl", and comment out the finite 
C        difference code
CML      _RL g_duIce(1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy)
CML      _RL g_dvIce(1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy)
CML      _RL g_uIceRes(1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy)
CML      _RL g_vIceRes(1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy)
CML
CMLC     Initialise
CML      DO bj=myByLo(myThid),myByHi(myThid)
CML       DO bi=myBxLo(myThid),myBxHi(myThid)
CML        DO J=1-Oly,sNy+Oly
CML         DO I=1-Olx,sNx+Olx
CML          g_duIce(I,J,bi,bj) = duice(I,J,bi,bj)
CML          g_dvIce(I,J,bi,bj) = dvice(I,J,bi,bj)
CML          g_uIceRes(I,J,bi,bj) = 0. _d 0
CML          g_vIceRes(I,J,bi,bj) = 0. _d 0
CML          uIceResP(I,J,bi,bj) = 0. _d 0
CML          vIceResP(I,J,bi,bj) = 0. _d 0
CML         ENDDO
CML        ENDDO
CML       ENDDO
CML      ENDDO
CML
CML      CALL G_SEAICE_CALC_RESIDUAL( uIce, g_duice, vIce, 
CML     $g_dvice, uiceresp, g_uiceres, viceresp, g_viceres, newtoniter, 
CML     $kryloviter, mytime, myiter, mythid )
CMLCML      For staf -v2 replace the above with the below call 
CMLCML      CALL SEAICE_CALC_RESIDUAL_TL( uIce, g_duice, vIce, 
CMLCML     $g_dvice, uiceresp, g_uiceres, viceresp, g_viceres, newtoniter, 
CMLCML     $kryloviter, mytime, myiter, mythid )
CML
CML      DO bj=myByLo(myThid),myByHi(myThid)
CML       DO bi=myBxLo(myThid),myBxHi(myThid)
CML        DO J=1-Oly,sNy+Oly
CML         DO I=1-Olx,sNx+Olx
CML          duice(I,J,bi,bj)=g_uiceres(I,J,bi,bj)
CML          dvice(I,J,bi,bj)=g_viceres(I,J,bi,bj)
CML         ENDDO
CML        ENDDO
CML       ENDDO
CML      ENDDO

C     Initialise
      epsilon = SEAICE_JFNKepsilon
      reps    = 1. _d 0/epsilon

      DO bj=myByLo(myThid),myByHi(myThid)
       DO bi=myBxLo(myThid),myBxHi(myThid)
        DO J=1-Oly,sNy+Oly
         DO I=1-Olx,sNx+Olx
          utp(I,J,bi,bj) = uIce(I,J,bi,bj) + epsilon * duIce(I,J,bi,bj)
          vtp(I,J,bi,bj) = vIce(I,J,bi,bj) + epsilon * dvIce(I,J,bi,bj)
         ENDDO
        ENDDO
       ENDDO
      ENDDO

C     Compute new residual F(u)
      CALL SEAICE_CALC_RESIDUAL(
     I     utp, vtp,
     O     uIceResP, vIceResP,
     I     newtonIter, krylovIter, myTime, myIter, myThid )

C     approximate Jacobian times vector by one-sided finite differences
C     and store in du/vIce
      DO bj = myByLo(myThid),myByHi(myThid)
       DO bi = myBxLo(myThid),myBxHi(myThid)
        DO I = 1, sNx
         DO J = 1, sNy
          duIce(I,J,bi,bj) = 
     &         (uIceResP(I,J,bi,bj)-uIceRes(I,J,bi,bj))*reps
          dvIce(I,J,bi,bj) =
     &         (vIceResP(I,J,bi,bj)-vIceRes(I,J,bi,bj))*reps
         ENDDO
        ENDDO
       ENDDO
      ENDDO

#endif /* SEAICE_ALLOW_JFNK */

      RETURN
      END
