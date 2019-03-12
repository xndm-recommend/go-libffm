package libffm

import (
	"bytes"
	"encoding/binary"
	"io/ioutil"
	"math"
)

type ffm_model struct {
	n             int // number of features
	m             int // number of fields
	k             int // number of latent factors
	w             []float32
	normalization bool
}

const (
	iFLOAT_LEN    int64 = 4
	iINT_LEN      int64 = 4
	iBOOL_LEN     int64 = 1
	iKALIGN       int64 = 4
	iKCHUNK_SIZE  int64 = 10000000
	iKMaxLineSize int64 = 100000
)

func minInt64(x, y int64) int64 {
	if x < y {
		return x
	}
	return y
}

func bytesToFloatList(b []byte) []float32 {
	var tmp []float32
	for i := int64(0); i < int64(len(b)); {
		tmp = append(tmp, bytesToFloat(b[i:i+iFLOAT_LEN]))
		i += iFLOAT_LEN
	}
	return tmp
}

func bytesToFloat(b []byte) float32 {
	bits := binary.LittleEndian.Uint32(b)
	return math.Float32frombits(bits)
}

func bytesToInt(b []byte) int {
	bits := binary.LittleEndian.Uint32(b)
	return int(bits)
}

func bytesToBool(b []byte) bool {
	var tmp bool
	binary.Read(bytes.NewBuffer(b), binary.LittleEndian, &tmp)
	return bool(tmp)
}

func get_k_aligned(k int) int {
	return int(math.Ceil(float64(k)/float64(iKALIGN)) * float64(iKALIGN))
}

func get_w_size(model ffm_model) int64 {
	k_aligned := get_k_aligned(model.k)
	return int64(model.n * model.m * k_aligned * 2)
}

func loadFfmModel(model_path string) (ffm_model, bool) {
	var model ffm_model
	var offset int64 = 0
	data, err := ioutil.ReadFile(model_path)
	if err != nil {
		return model, false
	}
	model.n = bytesToInt(data[offset : offset+iINT_LEN])
	offset += iINT_LEN
	model.m = bytesToInt(data[offset : offset+iINT_LEN])
	offset += iINT_LEN
	model.k = bytesToInt(data[offset : offset+iINT_LEN])
	offset += iINT_LEN
	model.normalization = bytesToBool(data[offset : offset+iBOOL_LEN])
	offset += iBOOL_LEN
	w_size := get_w_size(model)
	for i := int64(0); i < w_size; {
		next_offset := minInt64(w_size, int64(i+iFLOAT_LEN*iKCHUNK_SIZE))
		size := next_offset - i
		model.w = append(model.w, bytesToFloatList(data[offset+i:offset+i+iFLOAT_LEN*size])...)
		i = next_offset
	}
	return model, model.normalization
}
