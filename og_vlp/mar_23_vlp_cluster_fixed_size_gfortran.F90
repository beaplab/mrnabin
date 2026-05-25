! COMPILE: gfortran -cpp -O2 -fdec -o mar_23_vlp_cluster_fixed_size_gfortran mar_23_vlp_cluster_fixed_size_gfortran.F90
#define file_data        "sample_data/aa_bb_contigs.fasta"
! #define file_in3         "/home/alan/Run/Hd/3.5_cluster_all.txt"
#define file_in3         ""
! #define file_out         "/home/alan/Run/Hd/3.5_cluster_all.txt"
#define file_out         ""
! #define file_sym         "/home/alan/Run/HH/symbiant_hits.txt"
#define file_sym         ""

#define MAX_LEN_DATA           114000000
#define MAX_NUM_RECS              117000
#define MAX_SEQ_LEN                56000
#define MAX_NUM_STATES           2000000
#define POLY_THRESH                   50
#define CLUSTER_MIN_SEQ_LEN         3000
#define MIN_CLUSTER_SIZE               6
#define XCHI_THRESH                  255.0
! #define XCHI_THRESH                  293.0
#define THRESH_NEXT                    0
! #define THRESH_NEXT                  -10.0
#define XTHRESH_REMOVE                -3.0
#define KKIND 16
#define CKIND  1

#define LineLen              100
! #define print_it .true.
#define print_it .false.

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!! module  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

module vlp_stuff
  implicit integer(a-w,y,z)
  parameter( max_al = 4, nb = 3, nbr = 6, nc = 1 + (KKIND*8 - nbr) / nb)
!   parameter( max_al = 4, nb = 5, nbr = 5, nc = 1 + (KKIND*8 - nbr) / nb)
  parameter( max_num_states = MAX_NUM_STATES)
  real, allocatable :: mean(:), var(:), score(:, :, :), tmp_score(:, :)
  integer*CKIND, allocatable :: c(:), cc(:)
  integer           :: ac(0: 255), al, num_rec, num_long(0: 3), tot_train
  integer           :: len_thresh(0: 3) = (/1, 1000, 2000, 3000/)
  character*1       :: a(0 :127)
  integer*4, allocatable :: vlp_ptr(:, :)
  integer           :: ns
  real, allocatable  :: vlp_model(:, :)
  integer*KKIND, allocatable :: vlp_state(:)

  type record
     character*25   :: id
     integer        :: c_beg, c_end, c_len, cluster
     integer*1      :: direction, use_rec
     real           :: cnts(0: 255), percent
  end type record
  type(record), allocatable :: rec(:)

contains
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!! module subroutines !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!! make vlp_model !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  subroutine make_vlp_model(num_cluster)
    implicit integer(a-w,y,z)
    real    :: flat(0: max_al, - 1: nc)
    integer :: vlp_count(0: max_al, -1: nc - 1), org_count(0: max_al, -1: nc), level_state(-1: nc)
    integer :: f(0: 99)
    
    integer*KKIND, allocatable :: window(:), vlp2_state(:, :)
    integer*KKIND :: w, vmask(0: nc - 1), vmskshft(0: nc - 1)
    integer*KKIND :: state, new_state

    allocate(window(MAX_LEN_DATA), vlp2_state(2, MAX_NUM_STATES))
    vmask = 0
    forall(i = 1: nc - 1) vmask(i) = ishft(ibits(-1_16, 0, nb * i), nb * (nc - i))
    vmskshft = ishft(vmask, nbr - nb)
    if(print_it) print'(2i4,b130.128)', (i, popcnt(vmask(i)), vmask(i), i = 1, nc - 1)
    
! make  window of data and sort
    tot_train = 0
    do n = 1, num_rec
       if((rec(n)%cluster /= num_cluster) .or. (rec(n)%use_rec == 0)) cycle
       w = -1
       call reverse_complement(c(rec(n)%c_beg), cc, rec(n)%c_len,  rec(n)%direction)
       do t = 1, rec(n)%c_len                 ! store graph abcdefghijklmnopqrstuvwxy as xwvutsrqponmlkjihgfedcba.y
          tot_train = tot_train + 1           !             bcdefghijklmnopqrstuvwxyz    yxwvutsrqponmlkjihgfedcb.z
          window(tot_train) =  (ishft(ibits(w, nb * 2, nb * (nc - 2)), nb) .or. ishft(ibits(w, 0, nb), nb * (nc - 1))) .or. cc(t)
          w = window(tot_train)
       end do
    end do
    if(print_it) print"(i4,i2,x,a1,b130.128)", (t, c(t), a(c(t)), window(t), t = 1, 10)
    call shell_sort(tot_train, window)
    window(tot_train + 1) = -1
! count
    org_count = 0
    vlp_model = 0

    p = ibits(window(1), 0, nb)
    org_count(p, :) = org_count(p, :) + 1.0
 
    ns = 0
    pos_end = nb * (nc - 1)
    do t = 2, tot_train + 1
! find polygraphs - where do polygraphs diverge
       do k = 1, nc - 1
          if(((window(t) .neqv. window(t - 1)) .and. vmask(k)) .ne. 0_16) exit
       end do

       do lev = nc - 1, k, -1
          w = lev
          if(sum(org_count(0: al - 1, lev)) > POLY_THRESH) then
             flag = 1
             do pos = nb, pos_end, nb
                if( ibits(window(t - 1) .and. vmask(lev), pos, nb) >= al) flag = 0
             end do
             if(flag == 1) then
                ns = ns + 1
                vlp_model(0: al - 1, ns) =  org_count(0: al - 1, lev)
                vlp_state(ns) = ishft(window(t - 1) .and. vmask(lev), nbr - nb) .or. w
             end if
          end if
          org_count(:, lev) = 0
       end do

       if(t == (tot_train + 1)) exit
       p = ibits(window(t), 0, nb)
       org_count(p, :) = org_count(p, :) + 1
    end do
    ns = ns + 1
    vlp_state(ns) = 0_16
    vlp_model(0: al - 1, ns) =  org_count(0: al - 1, 0)
    
    if(print_it) print"('cluster: train_size - total',5i11)", num_cluster, count(rec(1:num_rec)%cluster == num_cluster), tot_train, ns, maxval(ibits(vlp_state(1:ns), 0, nbr))
  
    if(print_it) print "('number of states:'i10)", ns
    if(print_it) print "(b64.60,2i9/(10i11))", vlp_state(ns), tot_train, int(sum(vlp_model(:, ns))), int(vlp_model(:, ns))

! dump and flatten model - important to go backwards through polygraph list
    org_count = 0
    vlp_count = 0
    level_state = 0
    flat(0: al - 1, -1) = 1.0 / al
    vlp_model(al, :) = 1.0 / al
    old_lev = -1
    nz = 0
    do s = ns, 0, -1
       new_lev = 0
       if(s > 0) new_lev = ibits(vlp_state(s), 0, nbr)
       do lev = old_lev, new_lev, - 1
          ss = level_state(lev)

          xsum = max(sum(real(vlp_count(0: al - 1, lev))), 1.0)
          xlam = 1.0 / xsum
          vlp_model(0: al - 1, ss) = (1.0 - xlam) * vlp_count(0: al - 1, lev) / xsum + xlam * flat(0: al - 1, lev)
          xsum = sum(vlp_count(0: al - 1, lev))
          
       end do
       if(s < 1) exit
       
       old_lev = new_lev
       level_state(new_lev) = s
       org_count(0: al - 1, new_lev) = vlp_model(0: al - 1, s)
       vlp_count(0: al - 1, new_lev) = vlp_model(0: al - 1, s)
       
       xlam = 1.0 / sum(org_count(0: al - 1, new_lev))
       flat(0: al - 1, new_lev) = (1.0 - xlam) * org_count(0: al - 1, new_lev) / sum(org_count(0: al - 1, new_lev)) &
            & + xlam  *      flat(0: al - 1, new_lev - 1)
       
       xsum = sum(org_count(0: al - 1, new_lev))
       flat(0: al - 1, new_lev) = flat(0: al - 1, new_lev) / xsum
       vlp_model(0: al - 1, s)  = flat(0: al - 1, lev)
    end do
  
! make log wts
    vlp_model = max(vlp_model, 0.01)
    vlp_model(0: al, 1 :ns) = alog(al * vlp_model(0: al, 1 :ns)) / alog(2.0)
  
! connect model
! sort to set up binary search
    vlp2_state(1, 1: ns) = vlp_state(1: ns)
    forall(i = 1: ns) vlp2_state(2, i) = i
    call shell_sort2(ns, vlp2_state)

    vlp_ptr = 0
    vlp_ptr(al, :) = ns
    do s = 1, ns
       nlev = ibits(vlp_state(s), 0, nbr)
       state = iand(vlp_state(s), vmskshft(nlev))
       do k = 0, al - 1
          w = k
          new_state = ieor(ishft(state, -nb), ishft(w, pos_end + nbr - nb))
          do lev = min(nc - 1, nlev + 1), 0 , -1
             new_state = (iand(new_state, vmskshft(lev)) .xor. lev)
             j = ibinarysearch2(vlp2_state, ns, new_state)
             if(j == 0) cycle
             vlp_ptr(k, s) = vlp2_state(2, j)
             exit
          end do
       end do
    end do
    deallocate(window, vlp2_state)
    return
  end subroutine make_vlp_model

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!! make clusters  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  subroutine make_clusters(minlen, num_cluster)
    implicit integer(a-w,y,z)
    integer maxl(1)
    integer*1, allocatable :: ct(:, :)
    integer,   allocatable :: cluster(:), cluster_member(:)
    
    integer :: rev_map(0: 255)
    real :: tab(0:255, 2)
    
    rev_map = 0
    do p = 0, 6, 2
       forall(i = 0: 255) rev_map(i) = ishft(rev_map(i), 2) .or. ibits(255 - i, p, 2)
    end do

    ntot = count(rec%c_len >= minlen)
    print"(/'MAKING: clusters based on tetragraph counts',2i9)", minlen, ntot
    allocate(ct(ntot, ntot), cluster(ntot), cluster_member(ntot))  ! ct(i, j) = 1 => good chisq
    ct = 0
    n1 = 0
    do n = 1, num_rec
       if(rec(n)%c_len < minlen) cycle
       n1 = n1 + 1
       ct(n1, n1) = 1
       tab(:, 1) = rec(n)%cnts(:)
       n2 = n1
       do nn = n + 1, num_rec
          if(rec(nn)%c_len < minlen) cycle
          n2 = n2 + 1
          tab(:, 2) = rec(nn)%cnts(:)
          call ctcs(tab, 256, 2, dof, xchi0)
          tab(:, 2) = rec(nn)%cnts(rev_map(:))
          call ctcs(tab, 256, 2, dof, xchi1)
          xchi = min(xchi0, xchi1)
          if(xchi > XCHI_THRESH) cycle
          ct(n1, n2) = 1
          ct(n2, n1) = 1
       end do
    end do
! chain together to make clusters
    rec%cluster = 0
    num_cluster = 0
    do while(.true.)
       cluster = 0
       forall(n = 1: ntot) cluster(n) = sum(ct(:, n))
       maxl = maxloc(cluster)
       nmax = maxl(1)
       cluster = ct(:, nmax)
       if(sum(cluster) < 2) exit
       ct(:, nmax) = 0
       do while(.true.)
          old_sum = sum(cluster)
          do n = 1, ntot
             if( cluster(n) == 1) then
                cluster = cluster .or. ct(:, n)
                ct(:, n) = 0
             end if
          end do
          if(old_sum == sum(cluster)) exit
       end do

       if(sum(cluster) > 2) print"('cluster_density ',18x, 2i7)", num_cluster, sum(cluster)
       if(sum(cluster) < MIN_CLUSTER_SIZE) cycle
       num_cluster = num_cluster + 1
       cluster_member = 0
       nmem = 0
       n1 = 0
       do n = 1, num_rec
          if(rec(n)%c_len < minlen) cycle
          n1 = n1 + 1
          if(cluster(n1) == 1) then
             nmem= nmem + 1
             cluster_member(nmem) = n
             rec(n)%cluster = num_cluster
             rec(n)%use_rec = 1_1
          end if
       end do

       n = cluster_member(1)
       rec(n)%direction = 0
       tab(:, 1) = rec(n)%cnts(:)

       do while(.true.)
          change = 0
          do nn = 1, nmem
             n = cluster_member(nn)
             tab(:, 2) = rec(n)%cnts(:)
             call ctcs(tab, 256, 2, dof, xchi0)
             tab(:, 2) = rec(n)%cnts(rev_map(:))
             call ctcs(tab, 256, 2, dof, xchi1)
             direc= 0
             if(xchi1 < xchi0) direc = 1
             if( rec(n)%direction /= direc ) change = change + 1
             rec(n)%direction = direc
          end do
          if(change == 0) exit
          if(nmem >= MIN_CLUSTER_SIZE) print"(3i5,3x,100i1/(18x,100i1))", num_cluster, nmem, change, (rec(cluster_member(nn))%direction, nn = 1, nmem)
          tab = 0
          do nn = 1, nmem
             n = cluster_member(nn)
             if(rec(n)%direction == 0) tab(:, 1) = tab(:, 1) + rec(n)%cnts(:)
             if(rec(n)%direction == 1) tab(:, 1) = tab(:, 1) + rec(n)%cnts(rev_map(:))
          end do
       end do
       
       if(sum(cluster) >= MIN_CLUSTER_SIZE) then
          call make_vlp_model(num_cluster)
          print"('cluster_density ',18x, 4i7, 2i9)", num_cluster, sum(cluster), count(rec%cluster == num_cluster), maxval(ibits(vlp_state(1:ns), 0, nbr)), ns, tot_train
       end if
    end do
    print"('num_cluster',i5)", num_cluster
    deallocate(ct, cluster, cluster_member)
! stop
    return
  end subroutine make_clusters

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!! count tetragraphs !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  subroutine count_tetragraphs
    implicit integer(a-w,y,z)

    do n = 1, num_rec
       valid = 15
       tet = 0
       rec(n)%cnts = 0
       do t = rec(n)%c_beg, rec(n)%c_end
          tet = (ishft(tet, 2) .or. c(t)) .and. 255
          valid = ishft(valid, 1) .and. 15
          if(c(t) == al) valid = valid .or. 1
          if(valid == 0)  rec(n)%cnts(tet) = rec(n)%cnts(tet) + 1.0
       end do
    end do

    return
  end subroutine count_tetragraphs

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!! acgt_set !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  subroutine acgt_set
    implicit integer(a-w,y,z)
    character (len = 4)  :: b = 'ACGT'
    
    al = 4
    a = '.'
    ac = al
    do i = 1, al
       ac(ichar(b(i:i))) = i - 1
       a(i - 1) = b(i:i)
    end do

    return
  end subroutine acgt_set

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!! get_datam !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  subroutine get_datam
    implicit integer(a-w,y,z)
    character*1          :: ccline(LineLen)
    character*50         :: idm
    
! get records
    open(7, file = file_data)
    num_rec = 0
    rec(num_rec + 1)%c_beg = 1
    do while (.true.)
       read(7, *, IOSTAT = reason) idm
       if( reason < 0 ) exit
       if( idm(1:1) /= '>' ) cycle
       mlen = 0
       do while (.true.)
          read(7, "(100a1)", IOSTAT = reason) ccline
          if( reason < 0 ) exit
          if( ccline(1) == '>' ) then
             backspace 7
             exit
          end if
          do t = 1, LineLen
             if(ccline(t) == ' ') exit
             mlen = mlen + 1
             c(rec(num_rec + 1)%c_beg + mlen) = ac(ichar(ccline(t)))
          end do
       end do
       num_rec = num_rec + 1
       rec(num_rec)%id = trim(idm(2:))
!        if(index(idm, '_i', .True.) > 0) rec(num_rec)%id = idm(2:index(idm, '_i', .True.) - 1)
       rec(num_rec)%c_end     = rec(num_rec)%c_beg + mlen - 1
       rec(num_rec + 1)%c_beg = rec(num_rec)%c_end + 1
       rec(num_rec)%c_len     = mlen
       rec(num_rec)%cluster   = 0
       rec(num_rec)%direction = 0
       rec(num_rec)%use_rec   = 0
       if( reason < 0 ) exit
    end do
    close(7)
    print"('max_obs_seq_len:',i7,' total_num_obs',i10,'  num recs',i9)", maxval(rec(1:num_rec)%c_len), rec(num_rec)%c_end, num_rec

    if(file_sym == "") return
    rec%percent = 0.0
    open(7, file = file_sym)
    do while (.true.)
       read(7, "(a25, f8.2)", IOSTAT = reason) idm, xper
       if( reason < 0 ) exit
       idm = trim(idm)
       do n = 1, num_rec
          if(idm /= rec(n)%id) cycle
          rec(n)%percent = xper
          exit
       end do
    end do
    close(7)
    rec(1:num_rec)%use_rec = 1
    print"(20i6)", (count((rec%use_rec > 0) .and. (rec%cluster == i)                         ), i = 0, 9)
    print"(20i6)", (count((rec%use_rec > 0) .and. (rec%cluster == i) .and. (rec%percent > 0) ), i = 0, 9)
    rec(1:num_rec)%use_rec = 0

    return
  end subroutine get_datam
  

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!! vlp_xscore  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  real function vlp_xscore(c, cl, xvar)
    implicit integer(a-w,y,z)
    integer*CKIND :: c(cl)
  
    vlp_xscore = 0
    xvar = 0
    s = ns
    do t = 1, cl
       vlp_xscore = vlp_xscore + vlp_model(c(t), s)
       xvar = xvar + vlp_model(c(t), s)**2
       s = vlp_ptr(c(t), s)
    end do
    return 
  end function vlp_xscore

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!! reverse_complement !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  subroutine reverse_complement(c_in, c_out, cl, rev)
    implicit integer(a-w,y,z)
    integer*CKIND :: c_in(cl), c_out(cl), rev

    if( rev == 1) then
       forall(t = 1: cl)  c_out(cl + 1 - t) = 3 - c_in(t)
       where( c_out < 0 ) c_out = 4
    else
       c_out = c_in
    end if
    
    return
  end subroutine reverse_complement
  
end module vlp_stuff
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!! end module  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!! subroutines !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!!! ctcs    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
subroutine ctcs(tab, nr, nc, dof, xchi)
  implicit integer(a-w,y,z)
  real tab(nr, nc), rsum(nr), csum(nc), e(nr, nc)
  
  forall(i = 1:nr) rsum(i) = sum(tab(i, :))
  forall(j = 1:nc) csum(j) = sum(tab(:, j))
  tr = count(rsum > 0.5)
  tc = count(csum > 0.5)
  dof = (tr - 1) * (tc - 1)
  xtsum = sum(csum)
  xchi = 0
  do i = 1, nr
     if(rsum(i) < 0.5) cycle      
     do j = 1, nc
        if(csum(j) < 0.5) cycle
        xe = rsum(i) * csum(j) / xtsum
        xchi = xchi + (tab(i, j) - xe)**2 / max(0.01, xe)
     end do 
  end do    

  return
end subroutine ctcs

!!! argsort !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
subroutine argsort(n, r, d)
  integer r(n), d(n), il(n)
  
  integer :: stepsize
  integer :: i, j, n, left, k, ksize
  
  forall(i = 1:n) d(i) = i
  if ( n == 1 ) return
  
  stepsize = 1
  do while (stepsize < n)
     do left = 1 ,n - stepsize, stepsize * 2
        i = left
        j = left + stepsize
        ksize = min(stepsize * 2, n - left + 1)
        k=1
        
        do while ( i < left + stepsize .and. j < left + ksize )
           if ( r(d(i)) > r(d(j)) ) then
              il(k) = d(i)
              i = i + 1
              k = k + 1
           else
              il(k) = d(j)
              j = j + 1
              k = k + 1
           endif
        enddo
        
        if ( i < left + stepsize ) then  ! fill up remaining from left
           il(k: ksize) = d(i: left + stepsize - 1)
        else                             ! fill up remaining from right
           il(k: ksize) = d(j: left + ksize - 1)
        endif
        d(left: left + ksize - 1) = il(1: ksize)
     end do
     stepsize = stepsize * 2
  end do
  
  return
end subroutine argsort

!!! ibinarysearch2 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
integer function ibinarysearch2(x, n, val)
  
  implicit integer(a-z)
  integer*KKIND :: x(2, n), val
  
  start =  1
  finish = n
  range = finish - start
  mid = (start + finish) /2
  
  do while( (x(1, mid) .ne. val) .and. range >  0)
     if (val > x(1, mid)) then
        start = mid + 1
     else
        finish = mid - 1
     end if
     range = finish - start
     mid = (start + finish) / 2
  end do
  
  ibinarysearch2 = 0
  
  if( x(1, mid) .ne. val) then
     ibinarysearch2 = 0
  else
     ibinarysearch2 = mid
  end if

  return
  
end function ibinarysearch2

!!! shell_sort !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
subroutine shell_sort(n, a)
  integer*KKIND :: a(n), temp

  increment = n / 2
  do while (increment > 0)
      do i = increment + 1, n
         j = i
         temp = a(i)
         do while ((j >= (increment + 1)) .and. (a(j-increment) > temp))
            a(j) = a(j - increment)
            j = j - increment
            if((j - increment) < 1 ) exit
         end do
         a(j) = temp
      end do
      if (increment == 2) then
   	  increment = 1
      else
         increment = (increment * 5) / 11
      end if      
   end do
 
end subroutine shell_sort

!!! shell_sort2 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
subroutine shell_sort2(n, a)
  integer*KKIND :: a(2, n), temp(2)
 
  increment = n / 2
  do while (increment > 0)
      do i = increment + 1, n
         j = i
         temp = a(:, i)
         do while ((j >= increment + 1) .and. (a(1, j-increment) > temp(1)))
            a(:, j) = a(:, j - increment)
            j = j - increment
            if((j - increment) < 1 ) exit
         end do
         a(:, j) = temp
      end do
      if (increment == 2) then
   	  increment = 1
      else
         increment = (increment * 5) / 11
      end if      
   end do
 
 end subroutine shell_sort2


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!! compute_scores !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
subroutine compute_scores(ncl, iprint)
  use vlp_stuff
  integer, allocatable :: hist1(:), hist2(:) 
  
  if(iprint == 1) print"(/'BUILDING: VLP model')"
  score = -99999.0
  mean = 0
  var= 0
  do k = 1, ncl
     mtot = 0
     num_cluster = count(rec%cluster == k)
     if( num_cluster < 3) then
        where(rec%cluster == k) rec%cluster = 0
        cycle
     end if
     call make_vlp_model(k)
     do n = 1, num_rec
        if(rec(n)%use_rec == 0) cycle
        xsc0 = vlp_xscore(c(rec(n)%c_beg), rec(n)%c_len, xvar0)
        score(0, k, n) = xsc0
        call reverse_complement(c(rec(n)%c_beg), cc, rec(n)%c_len, 1_1)
        xsc1 =  vlp_xscore(cc, rec(n)%c_len, xvar1)
        score(1, k, n) = xsc1
        if(rec(n)%cluster == k) then
           mtot = mtot + rec(n)%c_len
           mean(k) = mean(k) + max(xsc0, xsc1)
           if(xsc1 > xsc0) then
              rec(n)%direction = 1
              var( k) = var( k) + xvar1
           else
              var( k) = var( k) + xvar0
              rec(n)%direction = 0
           end if
        end if
     end do
     mean(k) = mean(k) / mtot
     var( k) = (var(k) / mtot) - mean(k) * mean(k)
     if(iprint == 1) print"('cluster',3i7,3i9,2f9.5)", k, num_cluster, maxval(ibits(vlp_state(1:ns), 0, nbr)), ns, tot_train, mtot, mean(k), var(k)
     print"(i2,5i10,2f9.5)", k, count(rec%cluster == k), count((rec%cluster == k) .and. (rec%percent > 0)), maxval(ibits(vlp_state(1:ns), 0, nbr)), ns, mtot, mean(k), var(k)
     if(iprint == 0) cycle
     minv = -10
     maxv =  10
     allocate(hist1(minv: maxv), hist2(minv: maxv))
     hist1 = 0
     hist2 = 0
     do n = 1, num_rec
        if(rec(n)%use_rec == 0) cycle
        m = int( min(maxv, (max(minv, int((maxval(score(:,k, n)) - rec(n)%c_len * mean(k)) / sqrt(rec(n)%c_len * var(k)))) )) )
        if( k == rec(n)%cluster) hist1(m) = hist1(m) + 1
        if( k /= rec(n)%cluster) hist2(m) = hist2(m) + 1
     end do
     print"('Histogram_train: num_msg', i7,'  mean - var - sig',3f10.5)", count(rec%use_rec == 1), mean(k), var(k), sqrt(var(k))
     print"(25i6)", (i, i = minv, maxv)
     print"(25i6)", hist1, sum(hist1)
     print"(25i6)", hist2, sum(hist2)
     deallocate(hist1, hist2)
  end do
  where(rec%cluster > ncl) rec%cluster = 0
  return
end subroutine compute_scores

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!! find_best_scores !!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

subroutine find_best_scores(ncl, iprint)
  use vlp_stuff
  integer               :: maxl(1), maxl2(2), f(ncl)

  iter = 0
  ichange = 999
  print"('FIND: best scores. num clus',i3,'   recs in play',i9)", ncl, count(rec%use_rec > 0)
  print"('rec per clust',20i7)", (count((rec%cluster == i) .and. (rec%use_rec > 0)), i = 0, ncl)
  print"('sym per clust',20i7)", (count((rec%use_rec > 0) .and. (rec%cluster == i) .and. (rec%percent > 0) ), i = 0, ncl)
  forall(i = 1: ncl) f(i) = count(rec%cluster == i)
  do while(ichange > 0)
     iter = iter + 1
     call compute_scores(ncl, 0)
     ichange = 0
     do n = 1, num_rec
        if(rec(n)%use_rec == 0) cycle
        kc = rec(n)%cluster
        kd = rec(n)%direction
        maxl2 = maxloc(score(:,:, n))
        kcbest = maxl2(2)
        kdbest = maxl2(1)-1
        rec(n)%cluster   = kcbest
        rec(n)%direction = kdbest
        if((kc /= kcbest) .or. (kd /= kdbest)) ichange = ichange + 1
!         print"('change ',3i9,3i4,3f15.2,3x,a)", nn, n, rec(n)%c_len, k, kbest, kbest2, xbest_score, tmp_scores(kbest2), xbest_score - tmp_scores(kbest2), rec(n)%id
     end do
     xsum = sum(maxval(reshape(score, (/2*ncl, num_rec/)), 1),    MASK = rec%use_rec > 0)
     nz = count(maxval(reshape(score, (/2*ncl, num_rec/)), 1) < 0 .and. (rec%use_rec > 0))
     print"('ITER:',3i9,21x,'SCORE:',f15.2,32i9)", iter, ichange, count(f > 0), xsum, nz, ncl
  end do
  print"('rec per clust',20i7)", (count((rec%cluster == i) .and. (rec%use_rec > 0)), i = 0, ncl)
  print"('sym per clust',20i7)", (count((rec%use_rec > 0) .and. (rec%cluster == i) .and. (rec%percent > 0) ), i = 0, ncl)


  return
end subroutine find_best_scores

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!! remove_next_cluster !!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

subroutine remove_next_cluster(ncl, kbest, xbest_score)
  use vlp_stuff
  integer               :: maxl(1), cluster_tot(ncl)
  real                  :: cluster_score(ncl)

  call compute_scores(ncl, 1)
  cluster_tot   = 0
  cluster_score = 0

  do n = 1, num_rec
     if(rec(n)%use_rec == 0) cycle
     k = rec(n)%cluster
     cluster_tot(k) = cluster_tot(k) + 1
     tmp_score = score(:, :, n)
     tmp_score(:, k) = -99999.0
     cluster_score(k) = cluster_score(k) + maxval(tmp_score)
  end do
  where(cluster_tot == 0) cluster_score = -99999.
  cluster_score = cluster_score / max(cluster_tot, 1)
  maxl = maxloc(cluster_score)
  kbest = maxl(1)
  xbest_score = cluster_score(kbest)
  print"(/'REMOVE: next cluster'i9,i6,f9.2)", count(rec%use_rec > 0), kbest, xbest_score
  print"('rec per clust',20i7)", (count((rec%cluster == i) .and. (rec%use_rec > 0)), i = 0, ncl)
  print"('sym per clust',20i7)", (count((rec%use_rec > 0) .and. (rec%cluster == i) .and. (rec%percent > 0) ), i = 0, ncl)
  print"('remove clust ',20i7)", int(xbest_score), (int(cluster_score(i)), i = 1, ncl)
  if(xbest_score < THRESH_NEXT) return
  where(rec%cluster == kbest) rec%cluster = 0
  print"('rec per clust',20i7)", (count((rec%cluster == i) .and. (rec%use_rec > 0)), i = 0, ncl)
  print"('sym per clust',20i7)", (count((rec%use_rec > 0) .and. (rec%cluster == i) .and. (rec%percent > 0) ), i = 0, ncl)

  return
end subroutine remove_next_cluster



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!! main program !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

program tess
  use vlp_stuff
  implicit integer(a-w,y,z)
  integer      :: maxl(1)
  character*25 :: idc

  allocate(c(MAX_LEN_DATA), cc(MAX_SEQ_LEN))
  allocate(vlp_ptr(0: max_al, MAX_NUM_STATES))
  allocate(vlp_model(0: max_al, MAX_NUM_STATES))
  allocate(vlp_state(MAX_NUM_STATES))
  allocate(rec(MAX_NUM_RECS))

  call acgt_set
  call get_datam
  print"(2i8)", (v, count(rec(1:num_rec)%c_len >= v), v = 0, 10000, 500)

  if(file_in3 /= "") then
     open(9, file = file_in3)
     read(9, "(25x,2i3)") (rec(n)%cluster, rec(n)%direction, n = 1, num_rec)
     close(9)
     rec(1: num_rec)%use_rec = 1
     ncl = 16
     allocate(score(0:1, ncl, num_rec), tmp_score(0:1, ncl), mean(ncl), var(ncl))
     rec%use_rec = 0
     where(rec%c_len > 1000) rec%use_rec = 1
     print"('num_rec - sym',2i9)", num_rec, count(rec%percent > 0)
     print"('rec per clust',20i7)", (count((rec%cluster == i) .and. (rec%use_rec > 0)), i = 0, ncl)
     print"('sym per clust',20i7)", (count((rec%cluster == i) .and. (rec%use_rec > 0) .and. (rec%percent > 0)), i = 0, ncl)
     call compute_scores(ncl, 1)
     print"('num_rec - sym',2i9)", num_rec, count(rec%percent > 0)
     print"('rec per clust',20i7)", (count((rec%cluster == i) .and. (rec%use_rec > 0)), i = 0, ncl)
     print"('sym per clust',20i7)", (count((rec%cluster == i) .and. (rec%use_rec > 0) .and. (rec%percent > 0)), i = 0, ncl)
!      call find_best_scores(ncl, 1)
     print"('remove msg if score <',f6.2)", XTHRESH_REMOVE
     do n = 1, num_rec
        k = rec(n)%cluster
        if(k == 0) cycle
        xs = (score(rec(n)%direction, k, n) -(rec(n)%c_len * mean(k))) / sqrt(rec(n)%c_len * var(k))
        if(xs < XTHRESH_REMOVE) rec(n)%cluster = 0
     end do
!      call compute_scores(ncl, 1)
     print"('num_rec - sym',2i9)", num_rec, count(rec%percent > 0)
     print"('rec per clust',20i7)", (count((rec%cluster == i) .and. (rec%use_rec > 0)), i = 0, ncl)
     print"('sym per clust',20i7)", (count((rec%cluster == i) .and. (rec%use_rec > 0) .and. (rec%percent > 0)), i = 0, ncl)
     call find_best_scores(ncl, 1)
     print"('num_rec - sym',4i9)", num_rec, count(rec%percent > 0), count(rec%use_rec > 0), count((rec%use_rec > 0) .and. (rec%percent > 0))
     print"('rec per clust',20i7)", (count((rec%cluster == i) .and. (rec%use_rec > 0)), i = 0, ncl)
     print"('sym per clust',20i7)", (count((rec%cluster == i) .and. (rec%use_rec > 0) .and. (rec%percent > 0)), i = 0, ncl)
     stop
  end if
  
  
  call count_tetragraphs
  call make_clusters(CLUSTER_MIN_SEQ_LEN, ncl)
  print"('num_cluster',i6)", ncl
  allocate(score(0:1, ncl, num_rec), tmp_score(0:1, ncl), mean(ncl), var(ncl))
  call compute_scores(ncl, 1)
  call find_best_scores(ncl, 1)
  do nl = 3, 0, -1
     where(rec%c_len >= len_thresh(nl)) rec%use_rec = 1
     print"(/'Min_msg_len:'i7)", len_thresh(nl)
     call find_best_scores(ncl, 1)
     do while(.True.)
        call remove_next_cluster(ncl, kbest, xbest_score)
        if( xbest_score < THRESH_NEXT ) exit
        call find_best_scores(ncl, 1)
     end do
  end do

  if(file_out == "") stop
  
  open(9, file = file_out)
  write(9, "(a25,2i3,f11.2)") (rec(n)%id, rec(n)%cluster, rec(n)%direction, score(rec(n)%direction, rec(n)%cluster, n), n = 1, num_rec)
  close(9)

  stop
end program tess

