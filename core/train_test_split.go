package main

import (
	"bufio"
	"flag"
	"fmt"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/cheggaaa/pb"
)

type node struct {
	sccID   int
	visited bool
	edges   []int
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
		fmt.Println(fileName + " created")
		if err := fo.Close(); err != nil {
			panic(err)
		}
	}()
	output := ""
	for _, edge := range edges {
		start := min(edge.start,edge.end)
		end := max(edge.start, edge.end)
		output += strconv.Itoa(start) + " " + strconv.Itoa(end) + "\n"
	}
	output = output[:len(output)-1]
	fo.WriteString(output)
}

func min(x, y int) int {
    if x < y {
        return x
    }
    return y
}

func max(x, y int) int {
    if x > y {
        return x
    }
    return y
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
	n.sccID = -1
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
	g.nodes[h].edges = append(g.nodes[h].edges, t)
}

func (g *graph) removeEdge(t, h int) {
	if _, ok := g.nodes[t]; !ok {
		panic("No node for edge tail")
	}
	if _, ok := g.nodes[h]; !ok {
		panic("No node for edge head")
	}
	g.nodes[t].edges = removeFromSlice(g.nodes[t].edges, h)
	g.nodes[h].edges = removeFromSlice(g.nodes[h].edges, t)
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

func (g *graph) resetVisited() {
	for _, n := range g.nodes {
		n.visited = false
	}
}

func (g *graph) showGraph() {
	for k, v := range g.nodes {
		fmt.Printf("Node %d (SCC ID#: %d):\nEdges: %v\n\n", k, v.sccID, v.edges)
	}
}

func dfsMarkScc(n *node, g *graph, s int) {
	n.visited = true
	n.sccID = s
	for _, neighborLabel := range n.edges {
		if g.nodes[neighborLabel].visited == false {
			dfsMarkScc(g.nodes[neighborLabel], g, s)
		}
	}
}

func (g *graph) countUnvisited() int {
	count := 0
	for _, n := range g.nodes {
		if n.visited == false {
			count += 1
		}
	}
	return count
}

func (g *graph) findSccCount() int {
	g.resetVisited()
	s := 0
	for _, n := range g.nodes {
		if n.visited == false {
			dfsMarkScc(n, g, s)
			s++
		}
	}
	return s
}

func (g *graph) getAllEdges() []edgeType {
	var edges []edgeType
	for i, node := range g.nodes {
		for _, j := range node.edges {
			if i >= j {
				edges = append(edges, edgeType{j, i})
			}
		}
	}
	return edges
}

func (g *graph) readGraph() []edgeType {
	var edges []edgeType
	flag.Parse()
	if len(flag.Args()) < 2 {
		panic("Enter the name of the file with the graph edges list as first argument and \n Enter the directory to save train_edges_true and test_edges_true as the second argument")
	}
	f, err := os.Open(flag.Args()[0])
	if err != nil {
		panic(err)
	}
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := scanner.Text()
		iS := strings.Fields(line)
		if len(iS) != 2 {
			panic(fmt.Sprintf("Bad line in graph file: %v", iS))
		}
		t, err1 := strconv.Atoi(iS[0])
		if err1 != nil {
			panic(err1)
		}
		h, err2 := strconv.Atoi(iS[1])
		if err2 != nil {
			panic(err2)
		}
		g.addNode(t)
		g.addNode(h)
		g.addEdge(t, h)
		edges = append(edges, edgeType{t, h})
	}
	return edges
}

func makeTimestamp() int64 {
	return time.Now().UnixNano() / int64(time.Millisecond)
}

func batchEdgeRemovealSplittor(g *graph, batchEdges []edgeType, testEdges *[]edgeType) bool {

	flag := true
	for _, edge := range batchEdges {
		g.removeEdge(edge.start, edge.end)
	}

	ccc := g.findSccCount()
	if ccc > 1 {
		for _, edge := range batchEdges {
			g.addEdge(edge.start, edge.end)
		}
		flag = false
	} else {
		for _, edge := range batchEdges {
			*testEdges = append(*testEdges, edge)
		}
		flag = true
	}
	return flag
}

func split(g *graph, edges []edgeType, frac float32) []edgeType {
	batchSize := 64
	testCount := int(frac * float32(len(edges)))
	var testEdges []edgeType
	for {
		bar := pb.StartNew(len(edges) / batchSize)

		for i := 0; i < len(edges)-batchSize; i += batchSize {
			batch := edges[i : i+batchSize]
			status := batchEdgeRemovealSplittor(g, batch, &testEdges)
			if len(testEdges) > testCount {
				bar.Finish()
				break
			}
			// if rand.Float32() > 0.95 {
			// 	fmt.Println("End of batchSize", batchSize, ": ", len(testEdges), "/", testCount)
			// }

			if status {
				for j := 0; j < batchSize; j++ {
					edges[i+j] = edges[len(edges)-j-1]
				}
				edges = edges[:len(edges)-batchSize]
			}

			bar.Increment()
		}

		if len(testEdges) > testCount {
			bar.Finish()
			break
		}
		bar.Finish()
		fmt.Println("End of batchSize", batchSize, ": ", len(testEdges), "/", testCount)
		batchSize = int(batchSize / 2)
	}
	return testEdges
}

func main() {
	g := newGraph()
	edges := g.readGraph()
	fmt.Println(len(edges))
	rand.Seed(time.Now().UnixNano())
	for i := len(edges) - 1; i > 0; i-- { // Fisher–Yates shuffle
		j := rand.Intn(i + 1)
		edges[i], edges[j] = edges[j], edges[i]
	}
	testEdges := split(g, edges, 0.3)
	trainEdges := g.getAllEdges()
	fmt.Println(len(testEdges) + len(trainEdges))
	workDir := flag.Args()[1]
	if string(workDir[len(workDir)-1]) != "/"{
		workDir += "/"
	}
	writeToFile(testEdges, workDir + "test_edges_true.txt")
	writeToFile(trainEdges, workDir + "train_edges_true.txt")
	fmt.Println("Done")
}
