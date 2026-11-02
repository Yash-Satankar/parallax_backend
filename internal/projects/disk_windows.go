//go:build windows

package projects

import (
	"errors"
	"os"
	"syscall"
	"unsafe"
)

var (
	kernel32               = syscall.NewLazyDLL("kernel32.dll")
	procGetDiskFreeSpaceEx = kernel32.NewProc("GetDiskFreeSpaceExW")
)

const (
	errDiskFull       = syscall.Errno(112)
	errHandleDiskFull = syscall.Errno(39)
)

func freeBytes(path string) (int64, error) {
	if _, err := os.Stat(path); err != nil {
		if !os.IsNotExist(err) {
			return 0, err
		}
		path = "."
	}
	pathPtr, err := syscall.UTF16PtrFromString(path)
	if err != nil {
		return 0, err
	}
	var freeBytesAvailable, totalBytes, totalFreeBytes int64
	r1, _, callErr := procGetDiskFreeSpaceEx.Call(
		uintptr(unsafe.Pointer(pathPtr)),
		uintptr(unsafe.Pointer(&freeBytesAvailable)),
		uintptr(unsafe.Pointer(&totalBytes)),
		uintptr(unsafe.Pointer(&totalFreeBytes)),
	)
	if r1 == 0 {
		if callErr != nil && callErr != syscall.Errno(0) {
			return 0, callErr
		}
		return 0, errors.New("GetDiskFreeSpaceExW failed")
	}
	return freeBytesAvailable, nil
}

func isNoSpacePlatform(err error) bool {
	return errors.Is(err, errDiskFull) || errors.Is(err, errHandleDiskFull)
}
