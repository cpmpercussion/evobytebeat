// Bytebeat test program


main(t)
{
  for(;;t++) {
    putchar( (t<<1) * 2 + (t>>2) | 21^t >> t | t>>3 );
    //putchar(t);
  }
  // ((t<<1)^((t<<1)+(t>>7)&t>>12))|t>>(4-(1^7&(t>>19)))|t>>7);
}
