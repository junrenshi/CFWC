      subroutine metropolis_host(N, X, Y, XX, YY, M, d, a, STEPS,
     $     NACCEP)
      implicit none
      integer N, M, NMAX
      parameter(NMAX = 512)
      integer*8 NACCEP, STEPS, ns
      real*8 X(N), Y(N), XX(N), YY(N), p0save(NMAX), invrr2(NMAX, NMAX)
      double precision d, a, pi
      parameter (pi = 3.1415926535897932385d0)
      integer i, j, ic
      double precision genrand_real3, genrand_real1
      external genrand_real1, genrand_real3
      real*8 Xi, Yi, XXi, YYi, X0, Y0, X1, Y1, XXic, YYic, r2ij1, r2ij2
      double complex theta1p, z1, z0
      external theta1p
      double precision dx, dy,  r1, r2, r, p
      double precision p0, p1, c1, c2
      integer isave
      save p0save, invrr2,  c1, c2, isave
      data isave/0/

      if(isave .eq. 0)then
         c1 = pi / a
         c2 = dble(M) * (2.0d0 * pi/sqrt(3.0d0)/a**2)
         do i = 1, N
            dx = X(i) - XX(i)
            dy = Y(i) - YY(i)
            z1 = cmplx(dx, dy) * c1
            p0save(i) = abs(theta1p(z1))**(2*M) *
     $           exp(-(dx**2 + dy**2)*0.5d0
     $           + c2 * (dx**2-dy**2))
         end do
         
         do j = 1, N
            do i = 1, j-1
               invrr2(i, j) = 1.0d0
     $              /((XX(i)-XX(j))**2 + (YY(i)-YY(j))**2)**2
               invrr2(j, i) = invrr2(i, j)
            end do
            invrr2(j, j) = 1.0d0
         end do
         isave = 1
      end if

!      NACCEP = 0

      do ns = 1, steps

         ic = int(N * genrand_real3()) + 1

         X0 = X(ic)
         Y0 = Y(ic)

         r1 = genrand_real3()
         r2 = genrand_real1() * 2.0d0 * pi
         r1 = dsqrt(-2.0d0*dlog(r1))
         X1 = X0 + r1 * dcos(r2) * d
         Y1 = Y0 + r1 * dsin(r2) * d
         XXic = XX(ic)
         YYic = YY(ic)
 
!     Trick the Intel compiler to vectorize the loop
         X(ic) = XX(ic)
         Y(ic) = YY(ic)
         
         r1 = 1.0
         r2 = 1.0
         do i = 1, N
            Xi = X(i)
            XXi = XX(i)
            Yi = Y(i)
            YYi = YY(i)
            r1 = r1 *
     $           ((X1 - Xi) * (X1 - Xi) + (Y1 - Yi) * (Y1 - Yi))
            r1 = r1 * 
     $           ((X0 - XXi)*(X0 - XXi) + (Y0 - YYi)*(Y0 - YYi))            
            r1 = r1 * invrr2(i, ic)
            r2 = r2 *
     $           ((X1 - XXi)*(X1 - XXi) + (Y1 - YYi) * (Y1 - YYi))
            r2 = r2 * 
     $           ((X0 - Xi)*(X0 - Xi) + (Y0 -Yi)*(Y0 - Yi))
            r2 = r2 * invrr2(i, ic)
         end do
         X(ic) = X0
         Y(ic) = Y0
        
         p = (r1/r2)**M
         
         dx = X1 - XXic
         dy = Y1 - YYic
         z1 = cmplx(dx, dy) * c1
         p1 = abs(theta1p(z1))**(2*M) * 
     $        exp(-(dx**2 + dy**2) * 0.5d0 
     $        + c2 * (dx**2-dy**2))

         p0 = p0save(ic)

         p = p * (p1 / p0)

         if (p .ge. 1.0d0)then
            r = 0.0d0
         else
            r = genrand_real1()
         end if

         if(r .le. p)then
            X(ic) = X1
            Y(ic) = Y1
            NACCEP = NACCEP +1
            p0save(ic) = p1
         end if

      end do
      return
      end


      double complex function theta1p(z)
      implicit none
      double complex z, z1, t1, eiz, eiz2, eizinv, eiz2inv, ipitau, a2
      double precision pi, b2
      parameter (pi = 3.1415926535897932385d0)
!      parameter (ipitau = pi* cmplx(-sqrt(3.0d0) /2.0d0, 0.5d0))
!      parameter (b2 = 2.0d0/sqrt(3.0d0)/pi)
!      parameter (a2 = pi * cmplx(0.5d0, sqrt(3.0d0)/2.0d0))
      parameter (ipitau = (-2.720699046351327, 1.570796326794897))
      parameter (b2 = 0.3675525969478614)
      parameter (a2 = (1.570796326794897, 2.720699046351327))
      integer n, nz, nmax
      parameter(nmax = 7)
      double complex qn(nmax)
      data qn(1)/( 4.6797127951841172D-1,  1.9384005077761673D-1)/,
     $     qn(2)/(-2.0279163407481332D-3, -8.3999045169589555D-4)/,
     $     qn(3)/(-3.8081294567494075D-8, -1.5773788682580913D-8)/,
     $     qn(4)/( 3.098876062284498D-15,  1.283596493111571D-15)/,
     $     qn(5)/( 1.092766934009448D-24,  4.526388845795783D-25)/,
     $     qn(6)/(-1.669866367469883D-36, -6.916812967567200D-37)/,
     $     qn(7)/(-1.105774892572824D-50, -4.580269574353160D-51)/
      
      if(abs(z) .ge. 1d-8)then
         theta1p = dcmplx(0.0d0,0.0d0)
         t1 = dcmplx(1.0d0, 0.0d0)
      
         n = 0
         nz = imag(z) * b2
         if(nz .ne. 0)then
            z1 = z - nz * a2
         else
            z1 = z
         end if

         eiz = exp(dcmplx(0.0d0, 1.0d0) * z1)
         eiz2 = - eiz * eiz
         eizinv = 1.0d0 / eiz
         eiz2inv = - eizinv * eizinv

         do while (abs(t1) .gt. 1d-14
     $        .and. n .lt. nmax)
            t1 = qn(n+1) * (eiz - eizinv)
            theta1p = theta1p + t1
            eiz = eiz * eiz2
            eizinv = eizinv * eiz2inv
            n = n + 1
         end do
         theta1p = theta1p / z
         
         if(nz .ne. 0)then
            theta1p = theta1p * exp(-nz*nz * ipitau + 
     $           cmplx(0.0d0, -2.0d0*nz) * z1)
         end if
      else
         theta1p = dcmplx(-0.3927198865275040d0,0.9481096762683232d0)
      end if

      return 
      end



      subroutine CreateLattice(RC, RC0, a, Nc, No, N1, N2, XX, YY,
     $     Rotate, Reflect, Equiv)
      implicit none
      real*8 RC, RC0, a, XX(*), YY(*), X1, Y1, pi, eps
      parameter (pi = 3.1415926535897932385d0, eps=1d-6)
      integer Nc, No, N1, N2, N1o, N2o, Rotate(*), Reflect(*), Equiv(*)
      integer i, j, NMAX, imax, jmax
      parameter(NMAX = 512)
      integer bd(NMAX), bd1(NMAX)
      real*8  XX1(NMAX), YY1(NMAX)

!     Initializing the lattice
      Nc = 0
      No = 0
      imax = RC * 1.5
      jmax = sqrt(3.0d0) * imax
!     Elementary zone
      do i = 0, imax
         do j = 0, jmax
            X1 = dble(i) + dble(j)/2.0d0 
            Y1=  dble(j) * sqrt(3.0d0)/2.0d0
            
            if (sqrt(3.0d0)/2.0d0 * X1  + Y1 / 2.0d0 .LE. RC+eps
     $           .and.
     $           Y1 - 1.0d0/sqrt(3.0d0)*X1 .le. eps) then
               if(sqrt(3.0d0)/2.0d0 * X1  + Y1 / 2.0d0 .LE. RC0+eps)then
                  Nc = Nc + 1
                  XX(Nc) = X1 * a
                  YY(Nc) = Y1 * a
                  rotate(Nc) = 0
                  reflect(Nc) = 0
                  equiv(Nc) = Nc
                  if(Y1 .le. eps .or. Y1-1.0d0/sqrt(3.0d0)*X1 .ge. -eps)
     $                 then
                     bd(Nc) = 1
                  else
                     bd(Nc) = 0
                  end if
               else
                  No = No + 1
                  XX1(No) = X1 * a
                  YY1(No) = Y1 * a
                  if(Y1 .le. eps .or. Y1-1.0d0/sqrt(3.0d0)*X1 .ge. -eps)
     $                 then
                     bd1(No) = 1
                  else
                     bd1(No) = 0
                  end if
               end if
            end if
         end do
      end do

      N1 = Nc
      
!     Reflect over X-axis
      do i = 1, N1
         if(bd(i) .eq. 0)then
            Nc = Nc+1
            XX(Nc) = XX(i)
            YY(Nc) = - YY(i)
            equiv(Nc) = i
            reflect(Nc) = 1
            rotate(Nc) = 0
         end if
      end do
      
      N2 = Nc

      do j = 1, 5
         do i = 2, N2
            Nc = Nc+1
            XX(Nc) = XX(i) * cos(pi/3.0d0 * dble(j)) - YY(i) *
     $           sin(pi/3.0d0 * dble(j))
            YY(Nc) = XX(i) * sin(pi/3.0d0 * dble(j)) + YY(i) *
     $           cos(pi/3.0d0 * dble(j))
            equiv(Nc) = equiv(i)
            reflect(Nc) = reflect(i)
            rotate(Nc) = j
         end do
      end do

!     Setup outer sites
      N1o = No
      do i = 1, N1o
         if(bd1(i) .eq. 0)then
            No = No+1
            XX1(No) =   XX1(i)
            YY1(No) = - YY1(i)
         end if
      end do
      
      N2o = No

      do j = 1, 5
         do i = 1, N2o
            No = No+1
            XX1(No) = XX1(i) * cos(pi/3.0d0 * dble(j)) - YY1(i) *
     $           sin(pi/3.0d0 * dble(j))
            YY1(No) = XX1(i) * sin(pi/3.0d0 * dble(j)) + YY1(i) *
     $           cos(pi/3.0d0 * dble(j))
         end do
      end do

      do i = 1, No
         XX(Nc + i) = XX1(i)
         YY(Nc + i) = YY1(i)
      end do
      
      return
      end


      subroutine SymmetrizeAA(N, N1, N2, XX, YY, Rotate, Reflect, Equiv,
     $     AAXX, AAYY, AAXY, AAYX)
      implicit none
      integer N, N1, N2, Rotate(*), Reflect(*), Equiv(*)
      real*8 XX(*), YY(*)
      real*8 AAXX(*), AAYY(*), AAXY(*), AAYX(*), pi
      real*8 Ar, At, nx, ny, n1x, n1y, eps
      parameter (pi = 3.1415926535897932385d0)
      complex*16 zz, zzb
      integer i, j, k
      
 !     Symmetrize the correlation coefficients
      do i = 2, N2
         zz  = cmplx(AAXX(i) - AAYY(i), AAXY(i) + AAYX(i))
         zzb = cmplx(AAXX(i) + AAYY(i), AAXY(i) - AAYX(i))
         do j = 1, 5
            k = j*(N2-1) + i 
            zz = zz + cmplx(AAXX(k) - AAYY(k), AAXY(k) + AAYX(k)) * 
     $           exp(dcmplx(0.0d0, -2.0d0*pi/3.0d0 * dble(j)))
            zzb = zzb +  cmplx(AAXX(k) + AAYY(k), AAXY(k) - AAYX(k))
         end do
         zz  = zz / 6.0d0
         zzb = zzb / 6.0d0

         AAXX(i) = dble(zz + zzb)/2.0d0
         AAYY(i) = dble(zzb - zz)/2.0d0
         AAXY(i) = imag(zz + zzb)/2.0d0
         AAYX(i) = imag(zz - zzb)/2.0d0
      end do

      do i = N1 + 1, N2
         k = equiv(i)
         AAXX(k) = (AAXX(k) + AAXX(i))/2.0d0
         AAYY(k) = (AAYY(k) + AAYY(i))/2.0d0
         AAXY(k) = (AAXY(k) - AAXY(i))/2.0d0
         AAYX(k) = (AAYX(k) - AAYX(i))/2.0d0
         AAXX(i) = AAXX(k)
         AAYY(i) = AAYY(k)
         AAXY(i) = - AAXY(k)
         AAYX(i) = - AAYX(k)
      end do

      do i = N2+1, N
         k = equiv(i)
         zz  = cmplx(AAXX(k) - AAYY(k), AAXY(k) + AAYX(k))
         zzb = cmplx(AAXX(k) + AAYY(k), AAXY(k) - AAYX(k))
         if(reflect(i) .eq. 1)then
            zz = conjg(zz)
            zzb = conjg(zzb)
         end if

         zz = zz * exp(dcmplx(0, 2.0d0*pi/3.0d0*dble(rotate(i))))
         
         AAXX(i) = dble(zz + zzb)/2.0d0
         AAYY(i) = dble(zzb - zz)/2.0d0
         AAXY(i) = imag(zz + zzb)/2.0d0
         AAYX(i) = imag(zz - zzb)/2.0d0
      end do

!     On-site symmetrization
      AAXY(1) = 0.0d0
      AAYX(1) = 0.0d0
      AAXX(1) = (AAXX(1) + AAYY(1))*0.5d0
      AAYY(1) = AAXX(1)

      nx = 1.0
      ny = 0.0
      eps = 1d-6
      do j = 1, 6
         do i = 2, N
            if(abs(nx*YY(i) - ny*XX(i)) .LE. eps)then
               Ar = nx*AAXX(i)*nx + ny*AAYY(i)*ny + nx*AAXY(i)*ny
     $              + ny*AAYX(i)*nx
               At = ny*AAXX(i)*ny + nx*AAYY(i)*nx - ny*AAXY(i)*nx
     $              - nx*AAYX(i)*ny
               AAXX(i) = nx*nx * Ar + ny*ny * At
               AAYY(i) = ny*ny * Ar + nx*nx * At
               AAXY(i) = nx*ny * (Ar - At)
               AAYX(i) = AAXY(i)
            end if
         end do
         n1x = nx
         n1y = ny
         nx = cos(Pi/6.0d0) * n1x - sin(Pi/6.0d0) * n1y
         ny = sin(Pi/6.0d0) * n1x + cos(Pi/6.0d0) * n1y
      end do

      return
      end
      
     
      subroutine accuA(N, N0, X, Y, XX, YY, AXX, AYY, AXY, AYX)
      implicit none
      integer N, N0
      double precision X(N), Y(N), XX(N), YY(N), AXX(N), AYY(N), AXY(N),
     $     AYX(N)

      AXX = AXX + (X - XX) * (X(N0) - XX(N0))
      AYY = AYY + (Y - YY) * (Y(N0) - YY(N0))
      AXY = AXY + (X - XX) * (Y(N0) - YY(N0))
      AYX = AYX + (Y - YY) * (X(N0) - XX(N0))

      end


      subroutine accuAA(N, STEPS, AXX, AYY, AXY, AYX, 
     $     AAXX, AAYY, AAXY, AAYX, DAXX, DAYY, DAXY, DAYX)
      implicit none
      integer N
      integer*8 STEPS
      double precision AXX(N), AYY(N), AXY(N), AYX(N), AAXX(N), AAYY(N), 
     $     AAXY(N), AAYX(N), DAXX(N), DAYY(N), DAXY(N), DAYX(N)

      AAXX = AAXX + AXX / dble(STEPS)
      DAXX = DAXX + (AXX/dble(STEPS))**2

      AAYY = AAYY + AYY/dble(STEPS)
      DAYY = DAYY + (AYY/dble(STEPS))**2

      AAXY = AAXY +AXY/dble(STEPS)
      DAXY = DAXY + (AXY/dble(STEPS))**2

      AAYX = AAYX +AYX/dble(STEPS)
      DAYX = DAYX + (AYX/dble(STEPS))**2

      end

      
      
      subroutine normAA(N, NSTATS, AAXX, AAYY, AAXY, AAYX, 
     $     DAXX, DAYY, DAXY, DAYX)
      integer N, NSTATs
      double precision AAXX(N), AAYY(N), AAXY(N), AAYX(N),
     $     DAXX(N), DAYY(N), DAXY(N), DAYX(N)

      AAXX = AAXX / dble(NSTATS)
      DAXX = sqrt((DAXX/dble(NSTATS) - AAXX**2) /dble(NSTATS))

      AAYY = AAYY / dble(NSTATS)
      DAYY = sqrt((DAYY/dble(NSTATS) - AAYY**2) /dble(NSTATS))

      AAXY = AAXY / dble(NSTATS)
      DAXY = sqrt((DAXY/dble(NSTATS) - AAXY**2) /dble(NSTATS))

      AAYX = AAYX / dble(NSTATS)
      DAYX = sqrt((DAYX/dble(NSTATS) - AAYX**2) /dble(NSTATS))

      end      
