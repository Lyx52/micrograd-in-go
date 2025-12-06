package main

import (
	"fmt"
	"unsafe"
)

func main() {
	// 1. Create the value and the original, tracked pointer
	f := 3.14159
	trackedPtr := &f // The GC tracks this!

	// 2. Convert to uintptr for use in unsafe operations
	rawAddress := uintptr(unsafe.Pointer(trackedPtr))

	fmt.Printf("Original Ptr: %p\n", trackedPtr)
	fmt.Printf("Raw Address: 0x%x\n", rawAddress)

	// IMPORTANT: As long as 'trackedPtr' is in scope and reachable,
	// the GC will not free the memory for 'f'.

	// 3. You can safely convert the raw address back and use it
	//    because the memory is still protected by 'trackedPtr'.
	untrackedPtr := (*float64)(unsafe.Pointer(rawAddress))

	fmt.Printf("Value accessed via uintptr cast: %v\n", *untrackedPtr)
}
