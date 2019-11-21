package main

import (
	"fmt"
	"os"
	"strconv"
)

type node struct {
	edges []int
}

type edgeType struct {
	start int
	end   int
}

func writeToFile(edges []edgeType, fileName string) {
	fo, err := os.Create(fileName)
	if err != nil {
		panic(err)
	}
	defer func() {
		if err := fo.Close(); err != nil {
			panic(err)
		}
	}()
	output := ""
	for _, edge := range edges {
		output += strconv.Itoa(edge.start) + " " + strconv.Itoa(edge.end) + "\n"
	}
	output = output[:len(output)-1]
	fo.WriteString(output)
}

type graph struct {
	nodes map[int]*node
}

func newGraph() *graph {
	var g graph
	g.nodes = make(map[int]*node)
	return &g
}

func newNode() *node {
	var n node
	return &n
}

func (g *graph) addEdge(t, h int) {
	if _, ok := g.nodes[t]; !ok {
		panic("No node for edge tail")
	}
	if _, ok := g.nodes[h]; !ok {
		panic("No node for edge head")
	}
	g.nodes[t].edges = append(g.nodes[t].edges, h)
}

func removeFromSlice(slice []int, item int) []int {
	index := 0
	for idx, n := range slice {
		if n == item {
			index = idx
			break
		}
	}
	slice[index] = slice[len(slice)-1]
	return slice[:len(slice)-1]
}

func (g *graph) addNode(label int) bool {
	if _, ok := g.nodes[label]; !ok {
		n := newNode()
		g.nodes[label] = n
		return true
	}
	return false
}

func (g *graph) getAllEdges() []edgeType {
	var edges []edgeType
	for i, node := range g.nodes {
		for _, j := range node.edges {
			edges = append(edges, edgeType{j, i})
		}
	}
	return edges
}

func main() {
	g := newGraph()
	n, _ := strconv.Atoi(os.Args[1])
	clickSize, _ := strconv.Atoi(os.Args[2])
	for i := 0; i < clickSize; i++ {
		g.addNode(i)
		for j := i + 1; j < clickSize; j++ {
			g.addNode(j)
			g.addEdge(j, i)
		}
	}
	fmt.Println(len(g.getAllEdges()))
	clusterSize := int((n - clickSize) / clickSize)
	fmt.Println("clusterSize", clusterSize)
	for i := clickSize; i < n; i++ {
		connectTo := int((i - clickSize) / clusterSize)
		g.addNode(i)
		g.addEdge(i, connectTo)
	}
	fmt.Println(len(g.getAllEdges()))
	for _, e := range g.getAllEdges() {
		fmt.Println(e)
	}
}
