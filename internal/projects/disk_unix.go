//go:build !windows

package projects

import (
	"errors"
	"os"
	"syscall"
)

func freeBytes(path string) (int64, error) {
	if _, err := os.Stat(path); err != nil {
		if !os.IsNotExist(err) {
			return 0, err
		}
		path = "."
	}
	var st syscall.Statfs_t
	if err := syscall.Statfs(path, &st); err != nil {
		return 0, err
	}
	return int64(st.Bavail) * int64(st.Bsize), nil
}

func isNoSpacePlatform(err error) bool {
	return errors.Is(err, syscall.ENOSPC)
}
