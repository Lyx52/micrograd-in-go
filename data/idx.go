package data

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"os"

	"github.com/Lyx52/micrograd-in-go.git/nn"
)

const (
	UBYTE  = 0x08
	BYTE   = 0x09
	SHORT  = 0x0B
	INT    = 0x0C
	FLOAT  = 0x0D
	DOUBLE = 0x0E
)

type IdxHeader struct {
	Magic1    uint8
	Magic2    uint8
	DataType  uint8
	DimsCount uint8
}

func toGenericInt(arr []int32) []int {
	result := make([]int, len(arr))
	for i := range arr {
		result[i] = int(arr[i])
	}

	return result
}
func ReadIdx[TResult float64 | float32 | uint8 | int8 | int16 | int32](filePath string) ([]int, []TResult, error) {
	dims, result, err := ReadIdxUntyped(filePath)
	if err != nil {
		return nil, nil, err
	}
	validated, ok := result.([]TResult)
	if ok {
		return dims, validated, nil
	}

	return nil, nil, fmt.Errorf("invalid data type specified")
}
func ReadIdxUntyped(filePath string) ([]int, any, error) {
	file, err := os.Open(filePath)
	reader := bufio.NewReader(file)

	if err != nil {
		return nil, nil, err
	}

	header := IdxHeader{}
	err = binary.Read(reader, binary.BigEndian, &header)
	if err != nil {
		return nil, nil, err
	}

	if header.Magic1 != 0 || header.Magic2 != 0 {
		return nil, nil, fmt.Errorf("invalid magic numbers")
	}
	dims := make([]int32, header.DimsCount)

	err = binary.Read(reader, binary.BigEndian, &dims)
	if err != nil {
		return nil, nil, err
	}

	total := nn.GetTotalElements(dims)
	switch header.DataType {
	case UBYTE:
		result := make([]uint8, total)
		err = binary.Read(reader, binary.BigEndian, &result)
		return toGenericInt(dims), result, err
	case BYTE:
		result := make([]int8, total)
		err = binary.Read(reader, binary.BigEndian, &result)
		return toGenericInt(dims), result, err
	case DOUBLE:
		result := make([]float64, total)
		err = binary.Read(reader, binary.BigEndian, &result)
		return toGenericInt(dims), result, err
	case FLOAT:
		result := make([]float32, total)
		err = binary.Read(reader, binary.BigEndian, &result)
		return toGenericInt(dims), result, err
	case INT:
		result := make([]int32, total)
		err = binary.Read(reader, binary.BigEndian, &result)
		return toGenericInt(dims), result, err
	case SHORT:
		result := make([]int16, total)
		err = binary.Read(reader, binary.BigEndian, &result)
		return toGenericInt(dims), result, err
	}

	return nil, nil, fmt.Errorf("invalid data type %d", header.DataType)
}
